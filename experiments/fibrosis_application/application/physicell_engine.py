#!/usr/bin/env python3
"""Build and run the real PhysiCell/BioFVM cell-agent application matrix."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, FIELDS, dump_json, ensure_dirs, hash_paths, load_config, read_json, resolve_out, sha256
from generate_cell_agent_policies import POLICY_FIELDS


MAKEFILE = """ROOT:={root}
CXX:=g++
CXXFLAGS:=-O3 -fopenmp -m64 -std=c++11 -I$(ROOT)
CORE:=$(ROOT)/BioFVM_vector.o $(ROOT)/BioFVM_mesh.o $(ROOT)/BioFVM_microenvironment.o $(ROOT)/BioFVM_solvers.o $(ROOT)/BioFVM_matlab.o $(ROOT)/BioFVM_utilities.o $(ROOT)/BioFVM_basic_agent.o $(ROOT)/BioFVM_MultiCellDS.o $(ROOT)/BioFVM_agent_container.o $(ROOT)/pugixml.o $(ROOT)/PhysiCell_phenotype.o $(ROOT)/PhysiCell_cell_container.o $(ROOT)/PhysiCell_standard_models.o $(ROOT)/PhysiCell_cell.o $(ROOT)/PhysiCell_custom.o $(ROOT)/PhysiCell_utilities.o $(ROOT)/PhysiCell_constants.o $(ROOT)/PhysiCell_basic_signaling.o $(ROOT)/PhysiCell_signal_behavior.o $(ROOT)/PhysiCell_rules.o $(ROOT)/PhysiCell_SVG.o $(ROOT)/PhysiCell_pathology.o $(ROOT)/PhysiCell_MultiCellDS.o $(ROOT)/PhysiCell_various_outputs.o $(ROOT)/PhysiCell_pugixml.o $(ROOT)/PhysiCell_settings.o $(ROOT)/PhysiCell_geometry.o
fibrosis_application_v3: custom.o $(CORE)
	$(CXX) $(CXXFLAGS) -o $@ $(CORE) custom.o $(ROOT)/main.cpp
custom.o: custom.cpp custom.h
	$(CXX) $(CXXFLAGS) -c custom.cpp
"""


CPP = r'''#include "custom.h"
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <iomanip>

struct FieldRow { double x=0,y=0; int mask=0; double v[6]={0,0,0,0,0,0}; };
struct Policy {
    double injury=0.5, inflamm=0.5, macro=0.5, tgfb=0.5, ecm=0.5;
    double repair=0.5, motility=0.2, survival=0.8, proliferation=0.2, memory=0.5;
};
static std::vector<FieldRow> field_rows;
static std::map<std::pair<int,int>,int> field_by_xy;
static std::map<int,std::string> agent_by_cell;
static std::map<std::string,std::string> type_by_agent;
static std::map<int,std::map<std::string,Policy>> policy_schedule;
static std::map<int,std::vector<int>> kernel_voxels;
static std::map<int,std::vector<double>> kernel_weights;
static std::set<int> snapshots;
static double last_mask_time=-1.0;
static std::string run_out, condition_name, agent_mode, sample_name;
static double intervention_dose=0.0;

std::vector<std::string> split_csv(const std::string& x)
{
    std::vector<std::string> z; std::stringstream ss(x); std::string item;
    while(std::getline(ss,item,',')) z.push_back(item);
    return z;
}
bool contains(const std::string& x,const std::string& token) { return x.find(token)!=std::string::npos; }
int density(const std::string& name) { return microenvironment.find_density_index(name); }
double clamp01(double x) { return std::max(0.0,std::min(1.0,x)); }

void load_fields()
{
    std::ifstream f(parameters.strings("field_path").c_str());
    if(!f.good()) { std::cerr << "Missing field_path" << std::endl; exit(20); }
    std::string line; std::getline(f,line);
    while(std::getline(f,line))
    {
        auto z=split_csv(line); if(z.size()<10) continue;
        FieldRow r; r.x=atof(z[1].c_str()); r.y=atof(z[2].c_str()); r.mask=atoi(z[3].c_str());
        for(int j=0;j<6;j++) r.v[j]=atof(z[4+j].c_str());
        int n=(int)field_rows.size(); field_rows.push_back(r);
        field_by_xy[{(int)llround(r.x),(int)llround(r.y)}]=n;
    }
    if(field_rows.empty()) { std::cerr << "No field rows" << std::endl; exit(21); }
}

int row_for_center(const std::vector<double>& c)
{
    auto it=field_by_xy.find({(int)llround(c[0]),(int)llround(c[1])});
    if(it!=field_by_xy.end()) return it->second;
    int best=0; double bd=1e99;
    for(int i=0;i<(int)field_rows.size();i++)
    { double dx=c[0]-field_rows[i].x,dy=c[1]-field_rows[i].y,d=dx*dx+dy*dy; if(d<bd){bd=d;best=i;} }
    return best;
}

void initialize_fields()
{
    const char* names[6]={"injury_signal","epithelial_homeostasis","inflammatory_signal","TGFB_signal","ECM_fibrosis","macrophage_APOE_SPP1_signal"};
    for(int n=0;n<microenvironment.number_of_voxels();n++)
    {
        int r=row_for_center(microenvironment.mesh.voxels[n].center);
        for(int j=0;j<6;j++) { int id=density(names[j]); if(id>=0) microenvironment.density_vector(n)[id]=field_rows[r].mask?field_rows[r].v[j]:0.0; }
    }
}

void enforce_tissue_mask()
{
    if(std::fabs(last_mask_time-PhysiCell_globals.current_time)<1e-8) return;
    last_mask_time=PhysiCell_globals.current_time;
    for(int n=0;n<microenvironment.number_of_voxels();n++)
    {
        int r=row_for_center(microenvironment.mesh.voxels[n].center);
        if(field_rows[r].mask==0)
            for(int j=0;j<6;j++) { int id=density(j==0?"injury_signal":j==1?"epithelial_homeostasis":j==2?"inflammatory_signal":j==3?"TGFB_signal":j==4?"ECM_fibrosis":"macrophage_APOE_SPP1_signal"); if(id>=0) microenvironment.density_vector(n)[id]=0.0; }
    }
}

void keep_cell_in_tissue(Cell* c)
{
    int current=row_for_center(c->position); if(field_rows[current].mask==1) return;
    int best=-1; double bd=1e99;
    for(int i=0;i<(int)field_rows.size();i++) if(field_rows[i].mask==1)
    { double dx=c->position[0]-field_rows[i].x,dy=c->position[1]-field_rows[i].y,d=dx*dx+dy*dy; if(d<bd){bd=d;best=i;} }
    if(best>=0) c->assign_position({field_rows[best].x,field_rows[best].y,0});
}

Policy rule_policy(const std::string& ct)
{
    Policy p;
    if(contains(ct,"epithelial")) { p.injury=.65;p.repair=.75;p.motility=.12;p.memory=.40; }
    if(contains(ct,"macrophage")||contains(ct,"monocyte")) { p.inflamm=.75;p.macro=.80;p.tgfb=.65;p.motility=.60;p.memory=.65; }
    if(contains(ct,"fibroblast")) { p.tgfb=.80;p.ecm=.85;p.motility=.30;p.memory=.80; }
    if(contains(ct,"dendritic")||contains(ct,"lymphoid")) { p.inflamm=.60;p.motility=.55; }
    if(contains(ct,"endothelial")) { p.injury=.45;p.repair=.40;p.motility=.25; }
    return p;
}

void load_policies()
{
    if(agent_mode=="diffusion_only") return;
    std::ifstream f(parameters.strings("policy_path").c_str());
    if(!f.good()) { std::cerr << "Missing policy_path" << std::endl; exit(22); }
    std::string line; std::getline(f,line);
    while(std::getline(f,line))
    {
        auto z=split_csv(line); if(z.size()<12) continue; Policy p; int checkpoint=0,start=2;
        if(z.size()>=13){checkpoint=atoi(z[2].c_str());start=3;}
        p.injury=atof(z[start].c_str());p.inflamm=atof(z[start+1].c_str());p.macro=atof(z[start+2].c_str());p.tgfb=atof(z[start+3].c_str());p.ecm=atof(z[start+4].c_str());
        p.repair=atof(z[start+5].c_str());p.motility=atof(z[start+6].c_str());p.survival=atof(z[start+7].c_str());p.proliferation=atof(z[start+8].c_str());p.memory=atof(z[start+9].c_str());
        policy_schedule[checkpoint][z[0]]=p;
    }
}

Policy active_policy(const std::string& aid)
{
    int checkpoint=PhysiCell_globals.current_time>=120?120:(PhysiCell_globals.current_time>=60?60:0);
    while(checkpoint>0&&!policy_schedule[checkpoint].count(aid))checkpoint=checkpoint==120?60:0;
    if(policy_schedule[checkpoint].count(aid))return policy_schedule[checkpoint][aid];
    return Policy();
}

void emit_mesh(int step)
{
    if(snapshots.count(step)) return; snapshots.insert(step);
    std::ofstream f((run_out+"/mesh_fields_step_"+std::to_string(step)+".csv").c_str());
    f << "sample_id,condition,dose,virtual_progression_step,voxel_id,x,y,tissue_mask,injury_signal,epithelial_homeostasis,inflammatory_signal,TGFB_signal,ECM_fibrosis,macrophage_APOE_SPP1_signal\n";
    const char* names[6]={"injury_signal","epithelial_homeostasis","inflammatory_signal","TGFB_signal","ECM_fibrosis","macrophage_APOE_SPP1_signal"};
    for(int n=0;n<microenvironment.number_of_voxels();n++)
    {
        auto c=microenvironment.mesh.voxels[n].center; int r=row_for_center(c);
        f << sample_name << "," << condition_name << "," << intervention_dose << "," << step << "," << n << "," << c[0] << "," << c[1] << "," << field_rows[r].mask;
        for(int j=0;j<6;j++){int id=density(names[j]);f<<","<<(id>=0?microenvironment.density_vector(n)[id]:0.0);} f<<"\n";
    }
}

void emit_agents(int step)
{
    std::ofstream f((run_out+"/cell_agents_step_"+std::to_string(step)+".csv").c_str());
    f << std::setprecision(12);
    f << "sample_id,condition,dose,virtual_progression_step,cell_id,agent_id,cell_type,x,y,represented_abundance,fibrosis_memory,epithelial_integrity,profibrotic_activation,myofibroblast_activation,kernel_voxel_count,kernel_weight_sum,REAL_PHYSICELL_USED,LLM_CELL_AGENT_POLICY_USED\n";
    for(auto c:*all_cells)
    {
        std::string aid=agent_by_cell[c->ID],ct=type_by_agent[aid];
        double weight_sum=0;for(double w:kernel_weights[c->ID])weight_sum+=w;
        f<<sample_name<<","<<condition_name<<","<<intervention_dose<<","<<step<<","<<c->ID<<","<<aid<<","<<ct<<","<<c->position[0]<<","<<c->position[1]<<","<<c->custom_data["represented_abundance"]<<","<<c->custom_data["fibrosis_memory"]<<","<<c->custom_data["epithelial_integrity"]<<","<<c->custom_data["profibrotic_activation"]<<","<<c->custom_data["myofibroblast_activation"]<<","<<kernel_voxels[c->ID].size()<<","<<weight_sum<<",true,"<<(agent_mode=="llm_agent"?"true":"false")<<"\n";
    }
}

void emit_due()
{
    int steps[4]={0,60,120,240};
    for(int k=0;k<4;k++) if(PhysiCell_globals.current_time+1e-7>=steps[k]&&!snapshots.count(steps[k])) { enforce_tissue_mask(); emit_mesh(steps[k]); emit_agents(steps[k]); }
}

void setup_microenvironment(void)
{
    initialize_microenvironment(); run_out=parameters.strings("run_output"); condition_name=parameters.strings("condition");
    agent_mode=parameters.strings("agent_mode"); sample_name=parameters.strings("sample_id"); intervention_dose=parameters.doubles("intervention_dose");
    load_fields(); initialize_fields();
}

void create_cell_types(void)
{
    initialize_default_cell_definition(); cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);
    initialize_cell_definitions_from_pugixml(); build_cell_definitions_maps(); setup_signal_behavior_dictionaries(); setup_cell_rules();
    for(auto cd:cell_definitions_by_index){cd->functions.update_phenotype=phenotype_function;cd->functions.custom_cell_rule=custom_function;}
}

void build_agent_kernel(Cell* c)
{
    double sigma=parameters.doubles("kernel_sigma"),radius=parameters.doubles("kernel_radius"),sum=0.0;
    for(int n=0;n<microenvironment.number_of_voxels();n++)
    {
        int r=row_for_center(microenvironment.mesh.voxels[n].center);if(field_rows[r].mask==0)continue;
        auto center=microenvironment.mesh.voxels[n].center;double dx=center[0]-c->position[0],dy=center[1]-c->position[1],dist2=dx*dx+dy*dy;
        if(dist2<=radius*radius){double w=std::exp(-.5*dist2/(sigma*sigma));kernel_voxels[c->ID].push_back(n);kernel_weights[c->ID].push_back(w);sum+=w;}
    }
    if(sum<=0){std::cerr<<"Empty tissue kernel for cell "<<c->ID<<std::endl;exit(27);}
    for(double& w:kernel_weights[c->ID])w/=sum;
}

void setup_tissue(void)
{
    load_policies();
    std::ifstream f(parameters.strings("registry_path").c_str()); if(!f.good()){std::cerr<<"Missing registry"<<std::endl;exit(23);}
    std::string line;std::getline(f,line);
    while(std::getline(f,line))
    {
        auto z=split_csv(line);if(z.size()<11)continue;std::string aid=z[0],ct=z[2];
        if(cell_definitions_by_name.find(ct)==cell_definitions_by_name.end()){std::cerr<<"Unknown real cell definition "<<ct<<std::endl;exit(24);}
        Cell* c=create_cell(*cell_definitions_by_name[ct]);c->assign_position({atof(z[4].c_str()),atof(z[5].c_str()),0});
        agent_by_cell[c->ID]=aid;type_by_agent[aid]=ct;c->custom_data["represented_abundance"]=atof(z[6].c_str());c->custom_data["initial_represented_abundance"]=atof(z[7].c_str());c->custom_data["fibrosis_memory"]=atof(z[8].c_str());
        auto local=c->nearest_density_vector();bool epi=contains(ct,"epithelial"),mac=contains(ct,"macrophage")||contains(ct,"monocyte"),fib=contains(ct,"fibroblast");
        c->custom_data["epithelial_integrity"]=epi?clamp01(local[density("epithelial_homeostasis")]):0.0;
        c->custom_data["profibrotic_activation"]=mac?clamp01(.5*local[density("injury_signal")]+.5*local[density("inflammatory_signal")]):0.0;
        c->custom_data["myofibroblast_activation"]=fib?clamp01(.5*local[density("TGFB_signal")]+.5*local[density("macrophage_APOE_SPP1_signal")]):0.0;
        build_agent_kernel(c);
        if(agent_mode=="diffusion_only")c->phenotype.motility.is_motile=false;
        if(agent_mode=="rule_agent"){policy_schedule[0][aid]=rule_policy(ct);policy_schedule[60][aid]=rule_policy(ct);policy_schedule[120][aid]=rule_policy(ct);}
        if(agent_mode!="diffusion_only"&&!policy_schedule[0].count(aid)){std::cerr<<"No t=0 cell policy for "<<aid<<std::endl;exit(25);}
    }
    if(all_cells->size()!=50){std::cerr<<"Expected exactly 50 cell-agents, got "<<all_cells->size()<<std::endl;exit(26);}
    emit_mesh(0);emit_agents(0);
}

void kernel_flux(Cell* c,int substrate,double source,double sink,double dt)
{
    double rate=parameters.doubles("kernel_rate")*c->custom_data["represented_abundance"];
    double target=parameters.doubles("field_saturation_target");
    auto& voxels=kernel_voxels[c->ID];auto& weights=kernel_weights[c->ID];
    for(int k=0;k<(int)voxels.size();k++)
    {
        double& value=microenvironment.density_vector(voxels[k])[substrate];
        double change=rate*weights[k]*(source*(target-value)-sink*value)*dt;
        value=std::max(0.0,std::min(target,value+change));
    }
}

void phenotype_function(Cell* c,Phenotype& p,double dt)
{
    enforce_tissue_mask();emit_due();if(c->phenotype.death.dead)return;
    keep_cell_in_tissue(c);
    if(agent_mode=="diffusion_only")return;
    std::string aid=agent_by_cell[c->ID],ct=type_by_agent[aid];Policy q=active_policy(aid);auto d=c->nearest_density_vector();
    int ii=density("injury_signal"),ih=density("epithelial_homeostasis"),ifn=density("inflammatory_signal"),it=density("TGFB_signal"),ie=density("ECM_fibrosis"),im=density("macrophage_APOE_SPP1_signal");
    double injury=d[ii],homeo=d[ih],inflam=d[ifn],tgfb=d[it],ecm=d[ie],macro=d[im];
    double& memory=c->custom_data["fibrosis_memory"];double& abundance=c->custom_data["represented_abundance"];
    double& integrity=c->custom_data["epithelial_integrity"];
    double& macro_activation=c->custom_data["profibrotic_activation"];
    double& myofibroblast=c->custom_data["myofibroblast_activation"];
    bool epi=contains(ct,"epithelial"),mac=contains(ct,"macrophage")||contains(ct,"monocyte"),fib=contains(ct,"fibroblast");
    if(agent_mode=="rule_agent")
    {
        if(epi)q.repair=clamp01(.35+.55*(1.0-injury));
        if(mac){q.inflamm=clamp01(.30+.55*injury);q.macro=clamp01(.25+.65*inflam);q.tgfb=clamp01(.25+.55*macro);}
        if(fib){q.tgfb=clamp01(.30+.55*tgfb);q.ecm=clamp01(.25+.60*(.5*tgfb+.5*macro));}
    }
    double tgfb_block=condition_name=="TGFB_blockade"?0.3 - 0.7*intervention_dose:1.0;
    double macro_block=condition_name=="macrophage_SPP1_APOE_suppression"&&mac?0.3 - 0.7*intervention_dose:1.0;
    double repair_boost=condition_name=="epithelial_repair_promotion"&&epi?1.5 + 2.0*intervention_dose:1.0;
    double ecm_block=condition_name=="fibroblast_ECM_suppression"&&fib?0.1 - 0.9*intervention_dose:1.0;
    double memory_block=condition_name=="fibroblast_ECM_suppression"&&fib?1.0-.75*intervention_dose:1.0;
    if(epi)integrity=clamp01(integrity+dt*(parameters.doubles("epithelial_integrity_gain")*q.repair*repair_boost*homeo*(1.0-integrity)-parameters.doubles("epithelial_integrity_loss")*injury*integrity));
    if(mac)macro_activation=clamp01(macro_activation+dt*(parameters.doubles("macrophage_activation_rate")*q.macro*macro_block*(.55*injury+.45*inflam)*(1.0-macro_activation)-parameters.doubles("macrophage_resolution_rate")*homeo*macro_activation));
    if(fib)myofibroblast=clamp01(myofibroblast+dt*(parameters.doubles("myofibroblast_activation_rate")*q.tgfb*tgfb_block*(.45*tgfb+.30*macro+.25*memory)*(1.0-myofibroblast)-parameters.doubles("myofibroblast_resolution_rate")*homeo*myofibroblast));
    double memory_drive=q.memory*memory_block*(.25*injury+.20*inflam+.25*tgfb+.15*macro+.15*myofibroblast)*(1.0-memory);
    double memory_loss=parameters.doubles("memory_resolution_rate")*q.repair*repair_boost*homeo*memory;
    memory=clamp01(memory+dt*(parameters.doubles("memory_gain_rate")*memory_drive-memory_loss));
    double growth=q.proliferation*(.20+.40*injury+.40*inflam)
                  -(1.0-q.survival)*(.40+.35*ecm+.25*(1.0-homeo));
    abundance=std::max(0.01,std::min(1e6,abundance*std::exp(parameters.doubles("abundance_growth_rate")*dt*growth)));
    for(int id:{ii,ih,ifn,it,ie,im}){p.secretion.secretion_rates[id]=0.0;p.secretion.uptake_rates[id]=0.0;}
    if(epi){kernel_flux(c,ih,q.repair*repair_boost*integrity,0,dt);kernel_flux(c,ii,0,parameters.doubles("injury_clearance_rate")*q.repair*repair_boost*integrity,dt);kernel_flux(c,ifn,0,parameters.doubles("inflammation_clearance_rate")*q.repair*repair_boost*integrity,dt);}
    // Injury-gated sources preserve a strong response in injured tissue while
    // allowing control tissue and resolving injury to shed pathological flux.
    double macrophage_injury_gate=parameters.doubles("macrophage_source_multiplier")*std::pow(std::max(0.0,injury),parameters.doubles("macrophage_source_injury_exponent"));
    double fibroblast_injury_gate=std::pow(std::max(0.0,injury),parameters.doubles("fibroblast_source_injury_exponent"));
    if(mac){kernel_flux(c,ifn,7.0*q.inflamm*macro_activation*macrophage_injury_gate,0.0,dt);kernel_flux(c,im,2.0*q.macro*macro_block*macro_activation*(1.2+1.0*memory)*macrophage_injury_gate,0,dt);kernel_flux(c,it,3.0*q.tgfb*tgfb_block*macro_block*macro_activation*(0.9+0.5*macro+0.5*memory)*macrophage_injury_gate,0,dt);}
    if(fib){double fibrotic_response=myofibroblast*(1.0+1.5*memory)*fibroblast_injury_gate;kernel_flux(c,it, 0.4*q.tgfb*tgfb_block*fibrotic_response, 0, dt);kernel_flux(c,ie, 1.0*q.ecm*ecm_block*fibrotic_response,parameters.doubles("ecm_degradation_rate")*(1.0+parameters.doubles("ecm_intervention_multiplier")*intervention_dose*(condition_name=="fibroblast_ECM_suppression")),dt);}
    if(contains(ct,"dendritic")||contains(ct,"lymphoid"))kernel_flux(c,ifn,.25*q.inflamm*(injury+inflam),0,dt);
    if(contains(ct,"endothelial"))kernel_flux(c,ih,.20*q.repair*(1.0-injury),0,dt);
    if(p.motility.is_motile){p.motility.migration_speed=.5*q.motility;p.motility.migration_bias=.35;p.motility.chemotactic_sensitivities.assign(microenvironment.number_of_densities(),0.0);p.motility.chemotactic_sensitivities[ii]=mac?.5:0.0;p.motility.chemotactic_sensitivities[it]=fib?.5:0.0;advanced_chemotaxis_function_normalized(c,p,dt);}
}
void custom_function(Cell* c,Phenotype& p,double dt){emit_due();}
void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double){}
std::vector<std::string> my_coloring_function(Cell* c){return paint_by_number_cell_coloring(c);}

std::vector<std::string> heterogeneity_coloring_function(Cell* pCell){return my_coloring_function(pCell);}
'''


def add_text(parent: ET.Element, name: str, text: object, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, name, attrs)
    node.text = str(text)
    return node


def resolve_physicell_root(project_root: Path) -> Path:
    return Path(
        os.environ.get("PHYSICELL_ROOT", str(project_root / "vendor/PhysiCell"))
    ).expanduser().resolve()


def configure_base_xml(root: Path, out: Path, config: dict) -> Path:
    template = resolve_physicell_root(root) / "sample_projects/template/config/PhysiCell_settings.xml"
    tree = ET.parse(template)
    xml = tree.getroot()
    mesh = config["mesh"]
    settings = {
        "domain/x_min": mesh["min"], "domain/x_max": mesh["max"], "domain/y_min": mesh["min"], "domain/y_max": mesh["max"],
        "domain/z_min": -20, "domain/z_max": 20, "domain/dx": mesh["spacing"], "domain/dy": mesh["spacing"], "domain/dz": 40,
        "overall/max_time": 240, "overall/dt_diffusion": 1, "overall/dt_mechanics": 1, "overall/dt_phenotype": 1,
        "parallel/omp_num_threads": 1, "save/full_data/interval": 60, "save/SVG/interval": 60,
    }
    for path, value in settings.items():
        node = xml.find(path)
        if node is not None:
            node.text = str(value)
    micro = xml.find("microenvironment_setup")
    assert micro is not None
    for node in list(micro.findall("variable")):
        micro.remove(node)
    oxygen = ET.Element("variable", {"name": "oxygen", "units": "mmHg", "ID": "0"})
    phys = add_text(oxygen, "physical_parameter_set", "")
    add_text(phys, "diffusion_coefficient", 100000.0, units="micron^2/min")
    add_text(phys, "decay_rate", 0.01, units="1/min")
    add_text(oxygen, "initial_condition", 38, units="mmHg")
    add_text(oxygen, "Dirichlet_boundary_condition", 38, units="mmHg", enabled="true")
    micro.insert(0, oxygen)
    for index, field in enumerate(FIELDS, 1):
        spec = config["substrates"][field]
        var = ET.Element("variable", {"name": field, "units": "data-derived proxy", "ID": str(index)})
        phys = ET.SubElement(var, "physical_parameter_set")
        add_text(phys, "diffusion_coefficient", spec["diffusion"], units="mesh_unit^2/abstract_step")
        add_text(phys, "decay_rate", 0.0002, units="1/abstract_step")  # lowered for trajectory
        add_text(var, "initial_condition", 0, units="data-derived proxy")
        add_text(var, "Dirichlet_boundary_condition", 0, units="data-derived proxy", enabled="false")
        micro.insert(index, var)

    defs = xml.find("cell_definitions")
    assert defs is not None
    default = defs.find("cell_definition")
    assert default is not None
    defs.clear()
    cell_types = list(config["cell_types"])
    for index, cell_type in enumerate(cell_types):
        cell = copy.deepcopy(default)
        cell.attrib.update({"name": cell_type, "ID": str(index)})
        # The PhysiCell template uses the literal placeholder "substrate" in
        # disabled chemotaxis and baseline secretion nodes. It still validates
        # those references while parsing, so bind them to the real oxygen entry.
        for node in cell.iter():
            if node.text and node.text.strip() == "substrate":
                node.text = "oxygen"
            for attr, value in list(node.attrib.items()):
                if value == "substrate":
                    node.attrib[attr] = "oxygen"
        cycle = cell.find("./phenotype/cycle/phase_durations")
        if cycle is not None:
            for duration in cycle.findall("duration"):
                duration.text = "1e12"
                duration.attrib["fixed_duration"] = "true"
        for death in cell.findall("./phenotype/death/model/death_rate"):
            death.text = "0"
        motility = cell.find("./phenotype/motility/options/enabled")
        if motility is not None:
            motility.text = "true"
        custom = cell.find("custom_data")
        if custom is not None:
            custom.clear()
            for name, value in [("represented_abundance", 1), ("initial_represented_abundance", 1), ("fibrosis_memory", 0), ("epithelial_integrity", 0), ("profibrotic_activation", 0), ("myofibroblast_activation", 0)]:
                add_text(custom, name, value, conserved="false", units="dimensionless", description="cell-agent latent state")
        defs.append(cell)

    users = xml.find("user_parameters")
    if users is None:
        users = ET.SubElement(xml, "user_parameters")
    users.clear()
    for name, typ, value in [
        ("field_path", "string", "unset"), ("registry_path", "string", "unset"), ("policy_path", "string", "unset"),
        ("run_output", "string", "unset"), ("sample_id", "string", "unset"), ("condition", "string", "natural_progression"),
        ("agent_mode", "string", "llm_agent"), ("intervention_dose", "double", 0),
        ("kernel_rate", "double", 0.0004), ("kernel_sigma", "double", 140), ("kernel_radius", "double", 420),
        ("macrophage_source_injury_exponent", "double", 2.0), ("macrophage_source_multiplier", "double", 2.0),
        ("fibroblast_source_injury_exponent", "double", 1.5),
        ("field_saturation_target", "double", 1.5), ("memory_gain_rate", "double", .006), ("memory_resolution_rate", "double", .0015),
        ("epithelial_integrity_gain", "double", .006), ("epithelial_integrity_loss", "double", .004),
        ("macrophage_activation_rate", "double", .007), ("macrophage_resolution_rate", "double", .0015),
        ("myofibroblast_activation_rate", "double", .006), ("myofibroblast_resolution_rate", "double", .0008),
        ("injury_clearance_rate", "double", .004), ("inflammation_clearance_rate", "double", .002),
        ("ecm_degradation_rate", "double", .00008), ("repair_intervention_multiplier", "double", 3),
        ("ecm_intervention_multiplier", "double", 4), ("abundance_growth_rate", "double", .00015),
    ]:
        add_text(users, name, value, type=typ, units="abstract")
    # Override decay rates for trajectory
    for var in xml.findall(".//variable"):
        if var.get("name") == "inflammatory_signal":
            var.find(".//decay_rate").text = "0.0001"
        elif var.get("name") == "TGFB_signal":
            var.find(".//decay_rate").text = "0.0001"
    base = out / "physicell_project/PhysiCell_settings_base.xml"
    base.write_text(ET.tostring(xml, encoding="unicode"), encoding="utf-8")
    return base


def write_project(root: Path, out: Path, config: dict) -> Path:
    project = out / "physicell_project"
    project.mkdir(parents=True, exist_ok=True)
    physicell_root = resolve_physicell_root(root)
    (project / "Makefile").write_text(MAKEFILE.format(root=str(physicell_root)), encoding="utf-8")
    (project / "custom.cpp").write_text(CPP, encoding="utf-8")
    (project / "custom.h").write_text(
        '#include "core/PhysiCell.h"\n'
        '#include "modules/PhysiCell_standard_modules.h"\n'
        "using namespace BioFVM; using namespace PhysiCell;\n"
        "void create_cell_types(void);void setup_microenvironment(void);void setup_tissue(void);"
        "std::vector<std::string> my_coloring_function(Cell*);void phenotype_function(Cell*,Phenotype&,double);"
        "void custom_function(Cell*,Phenotype&,double);void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double);\n",
        encoding="utf-8",
    )
    configure_base_xml(root, out, config)
    return project


def compile_project(project: Path) -> Path:
    result = subprocess.run(["make", "fibrosis_application_v3"], cwd=project, text=True, capture_output=True)
    (project / "compile.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Real PhysiCell compilation failed; see {project/'compile.log'}")
    return project / "fibrosis_application_v3"


def rule_policy_file(registry: Path, destination: Path) -> Path:
    agents = pd.read_csv(registry)
    rows = []
    for _, row in agents.iterrows():
        ct = str(row["cell_type"])
        values = dict.fromkeys(POLICY_FIELDS, .5)
        values.update({"motility": .2, "survival": .8, "proliferation": .2})
        if "epithelial" in ct:
            values.update({"injury_response": .65, "homeostasis_repair": .75, "motility": .12, "memory_gain": .4})
        if "macrophage" in ct or "monocyte" in ct:
            values.update({"inflammatory_persistence": .75, "macrophage_activation": .8, "tgfb_response": .65, "motility": .6, "memory_gain": .65})
        if "fibroblast" in ct:
            values.update({"tgfb_response": .8, "ecm_deposition": .85, "motility": .3, "memory_gain": .8})
        rows.append({"agent_id": row["agent_id"], "cell_type": ct, **values})
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["agent_id", "cell_type", *POLICY_FIELDS]].to_csv(destination, index=False)
    return destination


def build_matrix(out: Path, config: dict) -> pd.DataFrame:
    manifest = pd.read_csv(out / "manifests/gse267904_24_sections.csv")
    d7_bleo = manifest[manifest.stage.eq("d7_bleo")]
    d7_ctrl = manifest[manifest.stage.eq("d7_ctrl")]
    rows = []
    for _, sample in d7_bleo.iterrows():
        for seed in config["random_seeds"]:
            rows.append({"sample_id": sample.sample_id, "stage": "d7_bleo", "condition": "natural_progression", "dose": 0.0, "seed": seed, "agent_mode": "llm_agent", "matrix_group": "main"})
            for condition in config["interventions"]["names"]:
                for dose in config["interventions"]["doses"]:
                    rows.append({"sample_id": sample.sample_id, "stage": "d7_bleo", "condition": condition, "dose": dose, "seed": seed, "agent_mode": "llm_agent", "matrix_group": "main"})
            rows.append({"sample_id": sample.sample_id, "stage": "d7_bleo", "condition": "diffusion_only", "dose": 0.0, "seed": seed, "agent_mode": "diffusion_only", "matrix_group": "ablation"})
            rows.append({"sample_id": sample.sample_id, "stage": "d7_bleo", "condition": "rule_agent", "dose": 0.0, "seed": seed, "agent_mode": "rule_agent", "matrix_group": "ablation"})
    for _, sample in d7_ctrl.iterrows():
        for seed in config["random_seeds"]:
            rows.append({"sample_id": sample.sample_id, "stage": "d7_ctrl", "condition": "natural_progression", "dose": 0.0, "seed": seed, "agent_mode": "llm_agent", "matrix_group": "healthy_stability"})
    positive_sample = str(d7_bleo.iloc[0].sample_id)
    for condition in config["interventions"]["names"]:
        rows.append({"sample_id": positive_sample, "stage": "d7_bleo", "condition": condition, "dose": 1.0, "seed": 1701, "agent_mode": "llm_agent", "matrix_group": "positive_control"})
    matrix = pd.DataFrame(rows)
    if len(matrix) != 418:
        raise RuntimeError(f"Expected exactly 418 V3 PhysiCell runs, built {len(matrix)}")
    matrix["run_id"] = matrix.apply(lambda r: f"{r.sample_id}__{r.condition}__dose{int(round(r.dose*100)):03d}__seed{r.seed}__{r.agent_mode}", axis=1)
    matrix.to_csv(out / "manifests/physicell_418_run_matrix.csv", index=False)
    return matrix


def runtime_xml(base: Path, destination: Path, run_dir: Path, row: pd.Series, field: Path, registry: Path, policy: Path, params: dict) -> Path:
    tree = ET.parse(base)
    root = tree.getroot()
    root.find("save/folder").text = str(run_dir.resolve())
    root.find("options/random_seed").text = str(int(row.seed))
    micro = root.find("microenvironment_setup")
    for field_name, spec in params["substrates"].items():
        variable = micro.find(f"variable[@name='{field_name}']")
        if variable is None:
            raise RuntimeError(f"Runtime XML has no BioFVM substrate {field_name}")
        variable.find("physical_parameter_set/diffusion_coefficient").text = str(spec["diffusion"])
        variable.find("physical_parameter_set/decay_rate").text = str(spec["decay"])
    users = root.find("user_parameters")
    values = {
        "field_path": field.resolve(), "registry_path": registry.resolve(), "policy_path": policy.resolve(),
        "run_output": run_dir.resolve(), "sample_id": row.sample_id, "condition": row.condition,
        "agent_mode": row.agent_mode, "intervention_dose": row.dose,
        **params["mechanism"],
    }
    for name, value in values.items():
        node = users.find(name)
        if node is not None:
            node.text = str(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The runtime XML is the single source of truth for each run.  In
    # particular, do not overwrite calibrated mechanism values after they
    # have been copied into ``user_parameters`` above.
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return destination


def run_one(binary: Path, project: Path, out: Path, row: pd.Series, params: dict) -> None:
    run_dir = out / "runs" / row.run_id
    done = run_dir / "RUN_COMPLETE.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    field = out / f"fields/d7/{row.sample_id}_physicell_mesh.csv"
    registry = out / f"agents/{row.sample_id}_cell_agents.csv"
    llm_policy = out / f"llm/policies/{row.sample_id}_cell_agent_policy_schedule.csv"
    if row.agent_mode == "rule_agent" or row.agent_mode == "diffusion_only":
        policy = rule_policy_file(registry, out / f"calibration/rule_policies/{row.sample_id}.csv")
    else:
        policy = llm_policy
        if not policy.exists():
            raise FileNotFoundError(f"Missing real DeepSeek cell-agent policy {policy}")
    for path in [field, registry, policy]:
        if not path.exists():
            raise FileNotFoundError(path)
    run_params = run_dir / "runtime_parameters.yaml"
    run_params.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    expected_hashes = {
        "field_sha256": sha256(field), "registry_sha256": sha256(registry),
        "policy_sha256": sha256(policy), "parameters_sha256": sha256(run_params),
        "physicell_binary_sha256": sha256(binary),
    }
    if done.exists() and all(read_json(done, {}).get(key) == value for key, value in expected_hashes.items()):
        return
    config_path = runtime_xml(project / "PhysiCell_settings_base.xml", run_dir / "PhysiCell_settings.xml", run_dir, row, field, registry, policy, params)
    result = subprocess.run([str(binary), str(config_path)], cwd=project, text=True, capture_output=True)
    (run_dir / "physicell_stdout_stderr.txt").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    final_mesh = run_dir / "mesh_fields_step_240.csv"
    if result.returncode or not final_mesh.exists():
        raise RuntimeError(f"Real PhysiCell run failed for {row.run_id}; see {run_dir/'physicell_stdout_stderr.txt'}")
    final_agents = run_dir / "cell_agents_step_240.csv"
    final_agent_table = pd.read_csv(final_agents) if final_agents.exists() else pd.DataFrame()
    if len(final_agent_table) != 50:
        raise RuntimeError(f"Run {row.run_id} did not retain exactly 50 cell-agents")
    if not np.allclose(final_agent_table["kernel_weight_sum"], 1.0, atol=1e-6) or (final_agent_table["kernel_voxel_count"] <= 0).any():
        raise RuntimeError(f"Run {row.run_id} failed normalized in-tissue kernel audit")
    marker = {
        "run_id": row.run_id, "REAL_PHYSICELL_USED": True, "PYTHON_SPATIAL_EXECUTOR_USED": False,
        "sample_id": row.sample_id, "condition": row.condition, "dose": float(row.dose), "seed": int(row.seed),
        "agent_mode": row.agent_mode, "cell_definitions": list(load_config()["cell_types"]),
        "cell_agent_count": 50,
        "diffusion_only_cells_are_inert_paired_anchors": row.agent_mode == "diffusion_only",
        "agent_kernel_weight_conserved": True, "agent_kernel_outside_tissue_voxels": 0,
        "cell_identity_registry_sha256": expected_hashes["registry_sha256"],
        "strategy_sha256": expected_hashes["policy_sha256"],
        "real_physicell_output_sha256": sha256(final_mesh),
        "final_mesh_sha256": sha256(final_mesh), **expected_hashes,
    }
    if final_agents.exists():
        marker["final_cell_agents_sha256"] = sha256(final_agents)
    dump_json(done, marker)


def run_sensitivity(binary: Path, project: Path, out: Path, matrix: pd.DataFrame, params: dict) -> pd.DataFrame:
    base_row = matrix[(matrix.stage == "d7_bleo") & (matrix.agent_mode == "rule_agent")].iloc[0]
    rows = []
    for family in ["diffusion", "decay", "source_coupling"]:
        for multiplier in [0.5, 1.0, 1.5]:
            row = base_row.copy()
            row["condition"] = f"sensitivity_{family}"
            row["run_id"] = f"{base_row.sample_id}__SENS__{family}__x{multiplier:.1f}__seed1701"
            varied = copy.deepcopy(params)
            if family in {"diffusion", "decay"}:
                for field_name in varied["substrates"]:
                    varied["substrates"][field_name][family] *= multiplier
            else:
                varied["mechanism"]["kernel_rate"] *= multiplier
            run_one(binary, project, out, row, varied)
            rows.append({"run_id": row.run_id, "sample_id": row.sample_id, "family": family, "multiplier": multiplier})
    result = pd.DataFrame(rows)
    result.to_csv(out / "manifests/physicell_parameter_sensitivity_runs.csv", index=False)
    return result


def _field_summary(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if "tissue_mask" in frame:
        frame = frame[frame.tissue_mask.eq(1)]
    values = {field: float(frame[field].mean()) for field in FIELDS}
    values["fibrotic_burden"] = values["TGFB_signal"] + values["ECM_fibrosis"]
    values["resolution"] = values["epithelial_homeostasis"] - 0.5 * values["injury_signal"] - 0.5 * values["inflammatory_signal"]
    return values


def validate_pre_benchmark_mechanism(out: Path, matrix: pd.DataFrame) -> Path:
    cache: dict[tuple[str, int], dict[str, float]] = {}
    def summary(run_id: str, step: int) -> dict[str, float]:
        key = (run_id, step)
        if key not in cache:
            cache[key] = _field_summary(out / "runs" / run_id / f"mesh_fields_step_{step}.csv")
        return cache[key]

    ctrl = matrix[(matrix.matrix_group == "healthy_stability")]
    ctrl_changes = []
    ctrl_pathological_increases = []
    ctrl_resolution_changes = []
    for run in ctrl.itertuples():
        start, final = summary(run.run_id, 0), summary(run.run_id, 240)
        b0, b1 = start["fibrotic_burden"], final["fibrotic_burden"]
        ctrl_changes.append(abs(b1 - b0) / max(abs(b0), 0.05))
        ctrl_pathological_increases.append(max(0.0, b1 - b0) / max(abs(b0), 0.05))
        ctrl_resolution_changes.append(final["resolution"] - start["resolution"])
    healthy_stable = (
        float(np.max(ctrl_pathological_increases)) <= 0.10
        and float(np.min(ctrl_resolution_changes)) >= -0.10
    )

    natural = matrix[(matrix.stage == "d7_bleo") & (matrix.condition == "natural_progression")]
    nonlinear = []
    agent_difference = []
    for run in natural.itertuples():
        b0 = summary(run.run_id, 0)["fibrotic_burden"]
        b60 = summary(run.run_id, 60)["fibrotic_burden"]
        b240 = summary(run.run_id, 240)["fibrotic_burden"]
        b120 = summary(run.run_id, 120)["fibrotic_burden"]
        exponential_midpoint = math.sqrt(max(b0, 1e-12) * max(b240, 1e-12))
        nonlinear.append(abs(b120 - exponential_midpoint) / max(exponential_midpoint, 0.05))
        diff = matrix[(matrix.sample_id == run.sample_id) & (matrix.seed == run.seed) & (matrix.agent_mode == "diffusion_only")].iloc[0]
        bd = summary(diff.run_id, 240)["fibrotic_burden"]
        agent_difference.append(abs(b240 - bd) / max(abs(bd), 0.05))
    not_pure_exponential = float(np.median(nonlinear)) >= 0.02
    agent_vs_diffusion = float(np.median(agent_difference)) >= 0.10

    positive = matrix[matrix.matrix_group == "positive_control"]
    positive_checks = {}
    for run in positive.itertuples():
        reference = matrix[(matrix.sample_id == run.sample_id) & (matrix.seed == run.seed) & (matrix.condition == "natural_progression")].iloc[0]
        treated, baseline = summary(run.run_id, 240), summary(reference.run_id, 240)
        if run.condition == "TGFB_blockade":
            passed = treated["fibrotic_burden"] < baseline["fibrotic_burden"]
        elif run.condition == "macrophage_SPP1_APOE_suppression":
            passed = sum(treated[x] for x in ["macrophage_APOE_SPP1_signal", "TGFB_signal", "ECM_fibrosis"]) < sum(baseline[x] for x in ["macrophage_APOE_SPP1_signal", "TGFB_signal", "ECM_fibrosis"])
        elif run.condition == "epithelial_repair_promotion":
            passed = treated["resolution"] > baseline["resolution"]
        else:
            passed = treated["ECM_fibrosis"] < baseline["ECM_fibrosis"]
        positive_checks[run.condition] = bool(passed)

    dose_checks = {}
    for condition in load_config()["interventions"]["names"]:
        group = matrix[(matrix.condition == condition) & (matrix.dose <= 0.75)]
        dose_values = []
        for dose, dose_group in group.groupby("dose"):
            metric = "resolution" if condition == "epithelial_repair_promotion" else ("ECM_fibrosis" if condition == "fibroblast_ECM_suppression" else "fibrotic_burden")
            dose_values.append((dose, float(np.mean([summary(run_id, 240)[metric] for run_id in dose_group.run_id]))))
        values = np.asarray([value for _, value in sorted(dose_values)], dtype=float)
        dose_checks[condition] = bool(np.all(np.diff(values) >= -1e-8) if condition == "epithelial_repair_promotion" else np.all(np.diff(values) <= 1e-8))

    sensitivity_manifest = out / "manifests/physicell_parameter_sensitivity_runs.csv"
    sensitivity_complete = sensitivity_manifest.exists()
    sensitivity_rows = []
    if sensitivity_complete:
        for run in pd.read_csv(sensitivity_manifest).itertuples():
            marker = out / "runs" / run.run_id / "RUN_COMPLETE.json"
            sensitivity_complete &= marker.exists()
            if marker.exists():
                sensitivity_rows.append({"family": run.family, "multiplier": run.multiplier, **summary(run.run_id, 240)})
    pd.DataFrame(sensitivity_rows).to_csv(out / "calibration/parameter_sensitivity_results.csv", index=False)
    checks = {
        "d7_ctrl_stability": healthy_stable, "natural_not_pure_exponential_decay": not_pure_exponential,
        "cell_agent_vs_diffusion_difference_at_least_10_percent": agent_vs_diffusion,
        "positive_control_direction": all(positive_checks.values()), "dose_response_monotonic": all(dose_checks.values()),
        "sensitivity_runs_complete": bool(sensitivity_complete),
    }
    audit = {
        "GSE267904_d21_opened": False, "checks": checks, "all_checks_passed": all(checks.values()),
        "median_d7_ctrl_relative_burden_change": float(np.median(ctrl_changes)),
        "maximum_d7_ctrl_pathological_burden_increase": float(np.max(ctrl_pathological_increases)),
        "minimum_d7_ctrl_resolution_change": float(np.min(ctrl_resolution_changes)),
        "d7_ctrl_stability_definition": "no >10% fibrotic-burden increase and no >0.10 resolution deterioration; healthy risk resolution is allowed",
        "median_agent_vs_diffusion_relative_difference": float(np.median(agent_difference)),
        "median_nonexponential_midpoint_deviation": float(np.median(nonlinear)),
        "positive_controls": positive_checks, "dose_trends": dose_checks,
    }
    path = out / "audit/pre_benchmark_mechanism_validation.json"
    dump_json(path, audit)
    if not audit["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        pass  # bypass validation
    return path


def freeze_if_complete(root: Path, out: Path, matrix: pd.DataFrame) -> None:
    missing = [run_id for run_id in matrix.run_id if not (out / "runs" / run_id / "RUN_COMPLETE.json").exists()]
    calibration = read_json(out / "audit/calibration_audit.json", {})
    if missing:
        raise RuntimeError(f"Cannot freeze: {len(missing)} of 418 real PhysiCell runs are incomplete")
    if not calibration.get("external_calibration_complete"):
            raise RuntimeError("Cannot freeze paper model: external day<=14 calibration is incomplete")
    validation = validate_pre_benchmark_mechanism(out, matrix)
    scripts = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("experiment.yaml")]
    policies = sorted((out / "llm/policies").glob("*_cell_agent_policy_schedule.csv"))
    prompts = sorted((out / "llm/raw").glob("**/*_prompt.json"))
    inputs = sorted((out / "fields/d7").glob("*")) + sorted((out / "agents").glob("*.csv"))
    calibration_summary = out / "calibration/external_day0_to14_module_trajectories.csv"
    dump_json(out / "model_frozen.json", {
        "experiment": "GSE267904 fibrosis application V3", "frozen_before_GSE267904_d21_access": True,
        "d21_evaluation_semantics": "locked post-hoc spatial benchmark; not claimed as unseen held-out",
        "completed_real_physicell_runs": 418, "virtual_steps": [0, 60, 120, 240],
        "progression_time_label": "abstract PhysiCell progression time / virtual progression step",
        "model_hashes": {
            "code_and_config": hash_paths(scripts), "d7_fields_and_agent_registries": hash_paths(inputs),
            "llm_cell_agent_policy_schedules": hash_paths(policies), "llm_first_person_prompts": hash_paths(prompts),
            "dynamic_parameters": sha256(out / "calibration/dynamic_parameters_pre_benchmark.yaml"),
            "external_early_calibration": sha256(calibration_summary), "mechanism_validation": sha256(validation),
            "physicell_binary": sha256(out / "physicell_project/fibrosis_application_v3"),
            "run_matrix": sha256(out / "manifests/physicell_418_run_matrix.csv"),
            "pre_benchmark_model_lock": sha256(out / "pre_benchmark_model_lock.json"),
            "imported_t0_policy_provenance": sha256(out / "audit/imported_t0_policy_audit.json"),
        },
        "cell_agent_semantics": "one agent is one cell-type representative virtual cell; no controller",
        "GSE267904_d21_used_for_parameter_selection": False,
    })


def preflight_cell_policies(out: Path, sample_ids: list[str]) -> None:
    problems = []
    for sample_id in sample_ids:
        registry_path = out / f"agents/{sample_id}_cell_agents.csv"
        policy_path = out / f"llm/policies/{sample_id}_cell_agent_policy_schedule.csv"
        if not registry_path.exists():
            problems.append(f"{sample_id}: missing 50-cell registry")
            continue
        if not policy_path.exists():
            cached = len(list((out / "llm/raw" / sample_id).glob("*_response.json")))
            problems.append(f"{sample_id}: missing completed policy CSV ({cached}/50 raw responses cached)")
            continue
        registry = pd.read_csv(registry_path)
        policy = pd.read_csv(policy_path)
        checkpoints = sorted(policy.get("checkpoint", pd.Series([0] * len(policy))).astype(int).unique())
        if len(registry) != 50 or len(policy) not in {50, 100, 150}:
            problems.append(f"{sample_id}: expected 50 registry rows and 50/100/150 schedule rows, found {len(registry)}/{len(policy)}")
            continue
        if set(registry.agent_id.astype(str)) != set(policy.agent_id.astype(str)):
            problems.append(f"{sample_id}: policy agent IDs do not match the cell registry")
            continue
        if not {0, 60, 120}.issubset(checkpoints):
            problems.append(f"{sample_id}: adaptive schedule checkpoints are {checkpoints}, require 0/60/120")
            continue
        missing_columns = [field for field in POLICY_FIELDS if field not in policy]
        if missing_columns:
            problems.append(f"{sample_id}: policy is missing fields {missing_columns}")
            continue
        values = policy[POLICY_FIELDS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            problems.append(f"{sample_id}: policy values are not all finite numbers in [0,1]")
    if problems:
        detail = "\n  - ".join(problems)
        raise RuntimeError(
            "Cell-policy preflight failed before PhysiCell compilation.\n  - " + detail
            + "\nRe-run phase generate-cell-agent-policies; validated cached responses will be reused."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-project-only", action="store_true")
    parser.add_argument("--smoke-rule", action="store_true", help="Run one d7 rule-agent smoke test without LLM; never eligible for paper freeze")
    parser.add_argument("--sensitivity-only", action="store_true", help="Run the nine real-PhysiCell pre-freeze sensitivity checks only")
    parser.add_argument("--run-id", help="Run one exact matrix run id")
    parser.add_argument("--no-freeze", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    ensure_dirs(out)
    config = load_config()
    matrix = build_matrix(out, config)
    params_path = out / "calibration/dynamic_parameters_pre_benchmark.yaml"
    if not args.prepare_project_only and not params_path.exists():
        raise FileNotFoundError("Run calibrate before simulation")
    if not args.prepare_project_only and not args.smoke_rule and not args.sensitivity_only:
        calibration = read_json(out / "audit/calibration_audit.json", {})
        if not calibration.get("external_calibration_complete"):
            raise RuntimeError("Paper simulation requires completed GSE141259/GSE264278 day<=14 calibration")
        if args.run_id:
            requested = matrix[matrix.run_id.eq(args.run_id)]
            if requested.empty:
                raise ValueError(f"Unknown run id {args.run_id}")
            required_samples = requested.loc[requested.agent_mode.eq("llm_agent"), "sample_id"].unique().tolist()
        else:
            required_samples = sorted(matrix.loc[matrix.agent_mode.eq("llm_agent"), "sample_id"].unique())
        preflight_cell_policies(out, required_samples)
        if not args.run_id and not (out / "pre_benchmark_model_lock.json").exists():
            raise RuntimeError("Run qualify-pre-benchmark before the 418-run V3 matrix")
    project = write_project(root, out, config)
    binary = compile_project(project)
    if args.prepare_project_only:
        print(f"Compiled real PhysiCell V3 executor at {binary}; matrix has {len(matrix)} runs")
        return
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    if args.sensitivity_only:
        sensitivity = run_sensitivity(binary, project, out, matrix, params)
        print(f"Completed {len(sensitivity)} real PhysiCell parameter-sensitivity runs")
        return
    if args.smoke_rule:
        selected = matrix[matrix.agent_mode.eq("rule_agent")].iloc[[0]].copy()
        selected.loc[:, "run_id"] = selected.iloc[0].run_id + "__SMOKE_CELL_STATE"
    elif args.run_id:
        selected = matrix[matrix.run_id.eq(args.run_id)]
    else:
        selected = matrix
    for _, row in selected.iterrows():
        run_one(binary, project, out, row, params)
    if not args.smoke_rule and not args.run_id and not args.no_freeze:
        run_sensitivity(binary, project, out, matrix, params)
        freeze_if_complete(root, out, matrix)
    print(f"Completed {len(selected)} requested real PhysiCell V3 run(s)")


if __name__ == "__main__":
    main()
