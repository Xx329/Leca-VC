#!/usr/bin/env python3
"""Generate, compile and run the independent real PhysiCell/BioFVM scenario."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1"
SCENARIO = ROOT / "scenarios/gse2565_bulk_macro_benchmark_v1"
PHYSICELL = Path(
    os.environ.get("PHYSICELL_ROOT", str(ROOT / "vendor/PhysiCell"))
).expanduser().resolve()
FIELDS = ["toxicant_proxy", "oxidative_stress", "epithelial_injury", "inflammation", "edema_proxy", "death_signal", "repair_signal"]


CPP = r'''
#include "custom.h"
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>

struct Policy { double oxidative=.1, injury=.1, inflammation=.1, edema=.1, death=.1, repair=.1, uptake=.1, motility=.1, sensitivity=.3, repair_response=.3; };
static std::map<int,std::string> worker_by_cell;
static std::map<int,std::string> agent_by_cell;
static std::map<int,std::map<std::string,Policy>> schedule;
static std::set<int> emitted;
static std::string run_out,condition_name;
static int expected_workers=0;

std::vector<std::string> split(const std::string& s){std::vector<std::string> z;std::stringstream ss(s);std::string x;while(std::getline(ss,x,','))z.push_back(x);return z;}
int density(const std::string& x){return microenvironment.find_density_index(x);}
double clamp01(double x){return std::max(0.0,std::min(1.0,x));}
bool contains(const std::string& x,const std::string& y){return x.find(y)!=std::string::npos;}

void load_policies(){
    std::ifstream f(parameters.strings("policy_path").c_str()); if(!f.good()){std::cerr<<"Missing policy file\n";exit(31);}
    std::string line;std::getline(f,line);auto h=split(line);std::map<std::string,int> col;for(int i=0;i<(int)h.size();i++)col[h[i]]=i;
    while(std::getline(f,line)){auto z=split(line);if(z.size()<h.size()||z[col["condition"]]!=condition_name)continue;Policy p;int minute=(int)std::llround(60.0*atof(z[col["checkpoint_hour"]].c_str()));
      p.oxidative=atof(z[col["oxidative_stress"]].c_str());p.injury=atof(z[col["epithelial_injury"]].c_str());p.inflammation=atof(z[col["inflammation"]].c_str());p.edema=atof(z[col["edema_proxy"]].c_str());p.death=atof(z[col["death_signal"]].c_str());p.repair=atof(z[col["repair_signal"]].c_str());p.uptake=atof(z[col["toxicant_uptake"]].c_str());p.motility=atof(z[col["motility"]].c_str());p.sensitivity=atof(z[col["injury_sensitivity"]].c_str());p.repair_response=atof(z[col["repair_response"]].c_str());schedule[minute][z[col["agent_id"]]]=p;}
}
Policy active(const std::string& agent){int m=PhysiCell_globals.current_time>=480?480:(PhysiCell_globals.current_time>=240?240:0);if(!schedule[m].count(agent)){std::cerr<<"Missing active policy for "<<agent<<" at "<<m<<"\n";exit(32);}return schedule[m][agent];}

void initialize_exposure(){int tox=density("toxicant_proxy");if(tox<0)return;bool injury=condition_name=="virtual_phosgene_injury";for(int n=0;n<microenvironment.number_of_voxels();n++){auto c=microenvironment.mesh.voxels[n].center;double r2=c[0]*c[0]+c[1]*c[1];microenvironment.density_vector(n)[tox]=injury?std::exp(-r2/(2.0*260.0*260.0)):0.0;}}

void emit(int minute){
  if(emitted.count(minute))return;emitted.insert(minute);
  std::ofstream w((run_out+"/work_cells_minute_"+std::to_string(minute)+".csv").c_str());w<<std::setprecision(12);
  w<<"minute,cell_id,worker_id,agent_id,x,y,represented_abundance,injury_state,inflammation_state,repair_state,active_policy_checkpoint,toxicant_proxy,oxidative_stress,epithelial_injury,inflammation,edema_proxy,death_signal,repair_signal,REAL_PHYSICELL_USED,AGENT_POLICY_APPLIED\n";
  for(auto c:*all_cells){std::string a=agent_by_cell[c->ID];int checkpoint=minute>=480?480:(minute>=240?240:0);auto d=c->nearest_density_vector();w<<minute<<","<<c->ID<<","<<worker_by_cell[c->ID]<<","<<a<<","<<c->position[0]<<","<<c->position[1]<<","<<c->custom_data["represented_abundance"]<<","<<c->custom_data["injury_state"]<<","<<c->custom_data["inflammation_state"]<<","<<c->custom_data["repair_state"]<<","<<checkpoint;const char* names[7]={"toxicant_proxy","oxidative_stress","epithelial_injury","inflammation","edema_proxy","death_signal","repair_signal"};for(int j=0;j<7;j++)w<<","<<d[density(names[j])];w<<",true,true\n";}
  std::ofstream m((run_out+"/substrates_minute_"+std::to_string(minute)+".csv").c_str());m<<std::setprecision(12)<<"minute,voxel_id,x,y,toxicant_proxy,oxidative_stress,epithelial_injury,inflammation,edema_proxy,death_signal,repair_signal\n";const char* names[7]={"toxicant_proxy","oxidative_stress","epithelial_injury","inflammation","edema_proxy","death_signal","repair_signal"};for(int n=0;n<microenvironment.number_of_voxels();n++){auto c=microenvironment.mesh.voxels[n].center;m<<minute<<","<<n<<","<<c[0]<<","<<c[1];for(int j=0;j<7;j++)m<<","<<microenvironment.density_vector(n)[density(names[j])];m<<"\n";}
}
void emit_due(){int steps[6]={0,30,60,240,480,720};for(int k=0;k<6;k++)if(PhysiCell_globals.current_time+1e-7>=steps[k])emit(steps[k]);}

void setup_microenvironment(void){initialize_microenvironment();run_out=parameters.strings("run_output");condition_name=parameters.strings("condition");initialize_exposure();}
void create_cell_types(void){initialize_default_cell_definition();cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);initialize_cell_definitions_from_pugixml();build_cell_definitions_maps();setup_signal_behavior_dictionaries();setup_cell_rules();for(auto cd:cell_definitions_by_index){cd->functions.update_phenotype=phenotype_function;cd->functions.custom_cell_rule=custom_function;}}
void setup_tissue(void){load_policies();std::ifstream f(parameters.strings("registry_path").c_str());if(!f.good()){std::cerr<<"Missing registry\n";exit(33);}std::string line;std::getline(f,line);while(std::getline(f,line)){auto z=split(line);if(z.size()<6)continue;std::string worker=z[0],agent=z[1],ct=z[2];if(!cell_definitions_by_name.count(ct)){std::cerr<<"Unknown cell type "<<ct<<"\n";exit(34);}Cell* c=create_cell(*cell_definitions_by_name[ct]);c->assign_position({atof(z[3].c_str()),atof(z[4].c_str()),0});worker_by_cell[c->ID]=worker;agent_by_cell[c->ID]=agent;c->custom_data["represented_abundance"]=atof(z[5].c_str());c->custom_data["injury_state"]=0;c->custom_data["inflammation_state"]=0;c->custom_data["repair_state"]=0;expected_workers++;}if(expected_workers!=parameters.ints("expected_workers")){std::cerr<<"Worker count mismatch\n";exit(35);}emit(0);}

void phenotype_function(Cell* c,Phenotype& p,double dt){emit_due();if(c->phenotype.death.dead)return;std::string a=agent_by_cell[c->ID];Policy q=active(a);auto d=c->nearest_density_vector();int tox=density("toxicant_proxy"),ox=density("oxidative_stress"),inj=density("epithelial_injury"),inf=density("inflammation"),ed=density("edema_proxy"),dea=density("death_signal"),rep=density("repair_signal");double local_tox=d[tox],local_inj=d[inj],local_inf=d[inf],local_rep=d[rep];double scale=c->custom_data["represented_abundance"]*expected_workers;bool epi=contains(a,"AT")||contains(a,"epithelial"),immune=contains(a,"macrophage")||contains(a,"monocyte")||contains(a,"neutrophil")||contains(a,"lymphoid"),endo=contains(a,"endothelial"),fib=contains(a,"fibroblast");p.secretion.uptake_rates[tox]=.002*q.uptake*scale;p.secretion.secretion_rates[ox]=.018*q.oxidative*(.05+local_tox)*scale;p.secretion.secretion_rates[inj]=.016*q.injury*(epi?1.0:.2)*(.05+local_tox+d[ox])*scale;p.secretion.secretion_rates[inf]=.014*q.inflammation*(immune?1.0:.25)*(.05+local_inj)*scale;p.secretion.secretion_rates[ed]=.012*q.edema*(endo?1.0:.15)*(.05+local_tox+local_inf)*scale;p.secretion.secretion_rates[dea]=.010*q.death*(epi||endo?1.0:.2)*(.05+local_inj+d[ox])*scale;p.secretion.secretion_rates[rep]=.010*q.repair*(fib||contains(a,"AT2")||contains(a,"resident_macrophage")?1.0:.25)*(.05+local_inj+local_inf)*scale;double &is=c->custom_data["injury_state"],&fs=c->custom_data["inflammation_state"],&rs=c->custom_data["repair_state"];is=clamp01(is+dt*.004*(q.sensitivity*(local_tox+d[ox]+local_inj)/3.0-is));fs=clamp01(fs+dt*.004*((local_inf+local_inj)/2.0-fs));rs=clamp01(rs+dt*.003*(q.repair_response*local_rep*(1.0-is)-rs));p.motility.migration_speed=.35*q.motility;p.motility.migration_bias=.0;}
void custom_function(Cell*,Phenotype&,double){emit_due();}
void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double){}
std::vector<std::string> my_coloring_function(Cell* c){return paint_by_number_cell_coloring(c);}
'''


MAKEFILE = r'''ROOT:={root}
CXX:=g++
CXXFLAGS:=-O2 -fopenmp -m64 -std=c++11 -I$(ROOT)
CORE:=$(ROOT)/BioFVM_vector.o $(ROOT)/BioFVM_mesh.o $(ROOT)/BioFVM_microenvironment.o $(ROOT)/BioFVM_solvers.o $(ROOT)/BioFVM_matlab.o $(ROOT)/BioFVM_utilities.o $(ROOT)/BioFVM_basic_agent.o $(ROOT)/BioFVM_MultiCellDS.o $(ROOT)/BioFVM_agent_container.o $(ROOT)/pugixml.o $(ROOT)/PhysiCell_phenotype.o $(ROOT)/PhysiCell_cell_container.o $(ROOT)/PhysiCell_standard_models.o $(ROOT)/PhysiCell_cell.o $(ROOT)/PhysiCell_custom.o $(ROOT)/PhysiCell_utilities.o $(ROOT)/PhysiCell_constants.o $(ROOT)/PhysiCell_basic_signaling.o $(ROOT)/PhysiCell_signal_behavior.o $(ROOT)/PhysiCell_rules.o $(ROOT)/PhysiCell_SVG.o $(ROOT)/PhysiCell_pathology.o $(ROOT)/PhysiCell_MultiCellDS.o $(ROOT)/PhysiCell_various_outputs.o $(ROOT)/PhysiCell_pugixml.o $(ROOT)/PhysiCell_settings.o $(ROOT)/PhysiCell_geometry.o
gse2565_bulk_macro_v1: custom.o $(CORE)
	$(CXX) $(CXXFLAGS) -o $@ $(CORE) custom.o $(ROOT)/main.cpp
custom.o: custom.cpp custom.h
	$(CXX) $(CXXFLAGS) -c custom.cpp
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def add(parent: ET.Element, name: str, value: object, **attrs: str) -> ET.Element:
    x = ET.SubElement(parent, name, attrs)
    x.text = str(value)
    return x


def generate_project(config: dict) -> Path:
    project = SCENARIO / "physicell_project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "custom.cpp").write_text(CPP)
    (project / "custom.h").write_text(f'#include "{PHYSICELL / "core/PhysiCell.h"}"\n#include "{PHYSICELL / "modules/PhysiCell_standard_modules.h"}"\nusing namespace BioFVM;using namespace PhysiCell;\nvoid create_cell_types(void);void setup_microenvironment(void);void setup_tissue(void);std::vector<std::string> my_coloring_function(Cell*);void phenotype_function(Cell*,Phenotype&,double);void custom_function(Cell*,Phenotype&,double);void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double);\n')
    (project / "Makefile").write_text(MAKEFILE.format(root=PHYSICELL.resolve()))
    template = PHYSICELL / "sample_projects/template/config/PhysiCell_settings.xml"
    tree = ET.parse(template)
    root = tree.getroot()
    d = config["domain"]
    values = {"domain/x_min":d["min"],"domain/x_max":d["max"],"domain/y_min":d["min"],"domain/y_max":d["max"],"domain/z_min":-10,"domain/z_max":10,"domain/dx":d["spacing"],"domain/dy":d["spacing"],"domain/dz":20,"overall/max_time":720,"overall/dt_diffusion":1,"overall/dt_mechanics":1,"overall/dt_phenotype":1,"parallel/omp_num_threads":1,"save/full_data/interval":30,"save/SVG/interval":30}
    for path, value in values.items():
        node=root.find(path)
        if node is not None: node.text=str(value)
    if root.find("save/full_data/enable") is not None: root.find("save/full_data/enable").text="true"
    if root.find("save/SVG/enable") is not None: root.find("save/SVG/enable").text="false"
    micro=root.find("microenvironment_setup")
    for node in list(micro): micro.remove(node)
    for i,name in enumerate(FIELDS):
        spec=config["substrates"][name];var=ET.SubElement(micro,"variable",{"name":name,"units":"normalized_proxy","ID":str(i)});phys=ET.SubElement(var,"physical_parameter_set");add(phys,"diffusion_coefficient",spec["diffusion"],units="micron^2/min");add(phys,"decay_rate",spec["decay"],units="1/min");add(var,"initial_condition",0,units="normalized_proxy");add(var,"Dirichlet_boundary_condition",0,units="normalized_proxy",enabled="false")
    defs=root.find("cell_definitions");default=defs.find("cell_definition");defs.clear()
    for i,name in enumerate(config["agent_types"]):
        cell=copy.deepcopy(default);cell.attrib.update({"name":name,"ID":str(i)})
        for node in cell.iter():
            if node.text and node.text.strip() in {"substrate","oxygen"}: node.text="toxicant_proxy"
            for key,value in list(node.attrib.items()):
                if value in {"substrate","oxygen"}: node.attrib[key]="toxicant_proxy"
        for death in cell.findall("./phenotype/death/model/death_rate"): death.text="0"
        for duration in cell.findall("./phenotype/cycle/phase_durations/duration"): duration.text="1e12";duration.attrib["fixed_duration"]="true"
        mot=cell.find("./phenotype/motility/options/enabled")
        if mot is not None: mot.text="true"
        custom=cell.find("custom_data")
        if custom is None: custom=ET.SubElement(cell,"custom_data")
        custom.clear()
        for key,value in [("represented_abundance",1),("injury_state",0),("inflammation_state",0),("repair_state",0)]: add(custom,key,value,conserved="false",units="dimensionless")
        defs.append(cell)
    users=root.find("user_parameters")
    if users is None: users=ET.SubElement(root,"user_parameters")
    users.clear()
    for name,typ,value in [("registry_path","string","unset"),("policy_path","string","unset"),("run_output","string","unset"),("condition","string","unset"),("expected_workers","int",50)]: add(users,name,value,type=typ,units="none")
    base=project/"PhysiCell_settings_base.xml";base.write_text(ET.tostring(root,encoding="unicode"))
    return project


def compile_project(project: Path) -> Path:
    subprocess.run(["make"],cwd=PHYSICELL,check=True,capture_output=True,text=True)
    result=subprocess.run(["make","gse2565_bulk_macro_v1"],cwd=project,capture_output=True,text=True)
    (SCENARIO/"compile.log").write_text(result.stdout+"\n"+result.stderr)
    if result.returncode: raise RuntimeError(f"PhysiCell compile failed: {SCENARIO/'compile.log'}")
    return project/"gse2565_bulk_macro_v1"


def runtime_xml(base: Path, dest: Path, run_dir: Path, registry: Path, policy: Path, condition: str, seed: int, max_hour: float) -> None:
    tree=ET.parse(base);root=tree.getroot();root.find("save/folder").text=str(run_dir.resolve());root.find("overall/max_time").text=str(max_hour*60);root.find("options/random_seed").text=str(seed);users=root.find("user_parameters");vals={"registry_path":registry.resolve(),"policy_path":policy.resolve(),"run_output":run_dir.resolve(),"condition":condition,"expected_workers":len(pd.read_csv(registry))}
    for key,value in vals.items(): users.find(key).text=str(value)
    dest.write_text(ET.tostring(root,encoding="unicode"))


def run_one(binary: Path, project: Path, model: str, condition: str, seed: int, max_hour: float) -> dict:
    run_id=f"{model}__{condition}__seed_{seed}__to_{max_hour:g}h";run_dir=OUT/"simulation"/run_id;run_dir.mkdir(parents=True,exist_ok=True)
    registry=OUT/"initialization/worker_registry.csv";policy=OUT/"policies"/("agent_policy_schedule.csv" if model=="agent_physicell_full" else "traditional_rule_policy_schedule.csv")
    cfg=run_dir/"PhysiCell_settings.xml";runtime_xml(project/"PhysiCell_settings_base.xml",cfg,run_dir,registry,policy,condition,seed,max_hour)
    result=subprocess.run([str(binary),str(cfg)],cwd=project,capture_output=True,text=True)
    (run_dir/"physicell_stdout_stderr.txt").write_text(result.stdout+"\n"+result.stderr)
    evidence=list(run_dir.glob("output*.xml"));expected=[0,30,60,240,480,720] if max_hour>=12 else [x for x in [0,30,60,240,480,720] if x<=max_hour*60]
    custom=[run_dir/f"work_cells_minute_{m}.csv" for m in expected]
    marker={"run_id":run_id,"model":model,"condition":condition,"seed":seed,"max_hour":max_hour,"exit_code":result.returncode,"REAL_PHYSICELL_USED":True,"PYTHON_SPATIAL_EXECUTOR_USED":False,"standard_output_xml_count":len(evidence),"required_work_cell_snapshots_present":all(x.exists() for x in custom),"binary_sha256":sha256(binary),"policy_sha256":sha256(policy),"registry_sha256":sha256(registry)}
    (run_dir/"run_audit.json").write_text(json.dumps(marker,indent=2)+"\n")
    if result.returncode or not evidence or not all(x.exists() for x in custom): raise RuntimeError(f"Real PhysiCell failed/incomplete for {run_id}")
    return marker


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","pilot","final"],required=True);a=p.parse_args();config=yaml.safe_load((HERE/"experiment.yaml").read_text());project=generate_project(config);binary=compile_project(project);seeds=config["seeds"][a.stage];max_hour=12 if a.stage in {"smoke","pilot"} else 72;rows=[]
    for seed in seeds:
        for model in config["dynamic_models"]:
            for condition in config["conditions"]: rows.append(run_one(binary,project,model,condition,int(seed),max_hour))
    manifest=OUT/"simulation"/f"{a.stage}_run_manifest.json";manifest.write_text(json.dumps(rows,indent=2)+"\n");print(json.dumps({"stage":a.stage,"runs":len(rows),"all_exit_zero":all(x["exit_code"]==0 for x in rows)}))


if __name__=="__main__": main()
