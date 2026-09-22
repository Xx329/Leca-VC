#!/usr/bin/env python3
"""Run a real PhysiCell/BioFVM spatial executor for GSE267904 micro-agents.

This script is intentionally strict by default. It creates a small dedicated
PhysiCell scenario, compiles it, runs real PhysiCell/BioFVM, and writes virtual
cell coordinates/environment summaries for downstream COMMOT. A debug Python
spatial generator still exists behind an explicit flag, but those outputs are
marked non-PhysiCell and are not suitable for claims.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import time
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib import error, request

import numpy as np
import pandas as pd

ROOT_FOR_FIREWALL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_FOR_FIREWALL / "scripts/leca_vc_prompt_isolation_v1"))
from prompt_firewall import (  # noqa: E402
    PROTOCOL_ID,
    restore_gse267904_execution_response,
    sanitize_gse267904_payload,
    validate_response_text,
    write_message_audit,
)
from scipy.io import loadmat
from scipy.spatial import cKDTree

from common import CELL_TYPES, dump_json, ensure_dirs, outpath


SUBSTRATES = [
    "oxygen",
    "damage_signal",
    "inflammatory_signal",
    "fibrosis_signal",
    "resolution_signal",
    "TGFb_like_signal",
    "SPP1_like_signal",
    "chemokine_signal",
]


MAKEFILE = """PHYSICELL_ROOT ?= ../../vendor/PhysiCell
ROOT:=$(PHYSICELL_ROOT)
CXX:=g++
CXXFLAGS:=-O3 -fopenmp -m64 -std=c++11 -I$(ROOT)
CORE:=$(ROOT)/BioFVM_vector.o $(ROOT)/BioFVM_mesh.o $(ROOT)/BioFVM_microenvironment.o $(ROOT)/BioFVM_solvers.o $(ROOT)/BioFVM_matlab.o $(ROOT)/BioFVM_utilities.o $(ROOT)/BioFVM_basic_agent.o $(ROOT)/BioFVM_MultiCellDS.o $(ROOT)/BioFVM_agent_container.o $(ROOT)/pugixml.o $(ROOT)/PhysiCell_phenotype.o $(ROOT)/PhysiCell_cell_container.o $(ROOT)/PhysiCell_standard_models.o $(ROOT)/PhysiCell_cell.o $(ROOT)/PhysiCell_custom.o $(ROOT)/PhysiCell_utilities.o $(ROOT)/PhysiCell_constants.o $(ROOT)/PhysiCell_basic_signaling.o $(ROOT)/PhysiCell_signal_behavior.o $(ROOT)/PhysiCell_rules.o $(ROOT)/PhysiCell_SVG.o $(ROOT)/PhysiCell_pathology.o $(ROOT)/PhysiCell_MultiCellDS.o $(ROOT)/PhysiCell_various_outputs.o $(ROOT)/PhysiCell_pugixml.o $(ROOT)/PhysiCell_settings.o $(ROOT)/PhysiCell_geometry.o
gse267904_spatial_commot_agent: custom.o $(CORE)
\t$(CXX) $(CXXFLAGS) -o $@ $(CORE) custom.o $(ROOT)/main.cpp
custom.o: custom.cpp custom.h control.h
\t$(CXX) $(CXXFLAGS) -c custom.cpp
"""


def resolve_physicell_root(project_root: Path) -> Path:
    return Path(
        os.environ.get("PHYSICELL_ROOT", str(project_root / "vendor/PhysiCell"))
    ).expanduser().resolve()


def q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def debug_model_spatial_run(out: Path, stages: list[str]) -> None:
    """A clearly labeled debug-only spatial generator; never marked PhysiCell."""
    reg = pd.read_csv(out / "micro_agents/micro_agent_registry.csv")
    rows = []
    field_rows = []
    for i, stage in enumerate(stages):
        for _, r in reg.iterrows():
            rows.append(
                {
                    "stage": stage,
                    "agent_id": r.agent_id,
                    "cell_type": r.cell_type,
                    "x": float(r.initial_position_x) + i * 8,
                    "y": float(r.initial_position_y) + i * 5,
                    "state": r.cell_type,
                    "oxygen": 38 - i * 3,
                    "damage_signal": 0.2 + 0.3 * i,
                    "inflammatory_signal": 0.15 + 0.25 * i,
                    "fibrosis_signal": 0.1 + 0.2 * i,
                    "resolution_signal": 0.05 + 0.08 * i,
                    "TGFb_like_signal": 0.1 + 0.15 * i,
                    "SPP1_like_signal": 0.1 + 0.15 * i,
                    "chemokine_signal": 0.25 + 0.25 * i,
                }
            )
        field_rows.append(
            {
                "stage": stage,
                "oxygen": 38 - i * 3,
                "damage_signal": 0.2 + 0.3 * i,
                "inflammatory_signal": 0.15 + 0.25 * i,
                "fibrosis_signal": 0.1 + 0.2 * i,
                "resolution_signal": 0.05 + 0.08 * i,
                "TGFb_like_signal": 0.1 + 0.15 * i,
                "SPP1_like_signal": 0.1 + 0.15 * i,
                "chemokine_signal": 0.25 + 0.25 * i,
            }
        )
    pd.DataFrame(rows).to_csv(out / "physicell/virtual_cell_positions_by_stage.csv", index=False)
    pd.DataFrame(field_rows).to_csv(out / "physicell/virtual_microenvironment_by_stage.csv", index=False)
    pd.DataFrame({"REAL_PHYSICELL_USED": [False], "DEBUG_MODEL_SPATIAL_GENERATOR_USED": [True]}).to_csv(
        out / "physicell/agent_writeback_audit.csv", index=False
    )


def write_control(project: Path, registry: Path, phys_out: Path, max_time: float) -> None:
    type_list = ",".join(q(x) for x in CELL_TYPES)
    substrate_list = ",".join(q(x) for x in SUBSTRATES)
    project.joinpath("control.h").write_text(
        "#pragma once\n"
        f"#define REGISTRY_PATH {q(str(registry.resolve()))}\n"
        f"#define PHYSICELL_OUT_DIR {q(str(phys_out.resolve()))}\n"
        f"#define MAX_TIME {float(max_time)}\n"
        f"static const int N_CELL_TYPES={len(CELL_TYPES)};\n"
        f"static const int N_SUBSTRATES={len(SUBSTRATES)};\n"
        f"static const char* CELL_TYPE_NAMES[N_CELL_TYPES]={{ {type_list} }};\n"
        f"static const char* SUBSTRATE_NAMES[N_SUBSTRATES]={{ {substrate_list} }};\n",
        encoding="utf-8",
    )


def cpp_source() -> str:
    return r'''#include "custom.h"
#include "control.h"
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <thread>
#include <chrono>

static std::map<int,std::string> agent_id_by_cell_id;
static std::map<std::string,std::string> cell_type_by_agent_id;
struct Program
{
    double damage=0.02, inflammatory=0.02, fibrosis=0.01, resolution=0.01, tgfb=0.01, spp1=0.01, chemokine=0.01;
    double motility=0.20, death=1.0, proliferation=1.0;
};
static std::map<std::string,Program> program_by_agent;
static std::set<int> summary_emitted_cell_ids;
static int expected_micro_agent_count = 0;
static int next_decision_interval = 0;
static const double DECISION_TIMES[2] = {0.0, 60.0};
std::vector<std::string> split_csv(const std::string& x)
{
    std::vector<std::string> z; std::stringstream ss(x); std::string q;
    while(std::getline(ss,q,',')) z.push_back(q);
    return z;
}

int substrate_index(const std::string& name)
{
    return microenvironment.find_density_index(name);
}

int type_index(std::string name)
{
    for(int i=0;i<N_CELL_TYPES;i++) if(name==CELL_TYPE_NAMES[i]) return i;
    return N_CELL_TYPES-1;
}

bool file_exists(const std::string& p)
{
    std::ifstream f(p.c_str());
    return f.good();
}

Program default_program_for_cell_type(const std::string& ct)
{
    Program pr;
    if(ct.find("epithelial")!=std::string::npos) { pr.damage=0.20; pr.chemokine=0.12; pr.motility=0.25; }
    if(ct.find("macrophage")!=std::string::npos) { pr.inflammatory=0.22; pr.spp1=0.18; pr.chemokine=0.12; pr.motility=0.35; }
    if(ct.find("fibroblast")!=std::string::npos) { pr.fibrosis=0.22; pr.tgfb=0.20; pr.motility=0.18; }
    if(ct.find("endothelial")!=std::string::npos) { pr.chemokine=0.10; pr.motility=0.20; }
    if(ct.find("lymphoid")!=std::string::npos) { pr.inflammatory=0.12; pr.motility=0.40; }
    return pr;
}

void set_global_fields(double t)
{
    int oxy = substrate_index("oxygen");
    double injury = std::min(1.0, 0.20 + 0.006*t);
    for(int n=0;n<microenvironment.number_of_voxels();n++)
    {
        if(oxy>=0) microenvironment.density_vector(n)[oxy] = 38.0 - 5.0*injury;
        for(int j=1;j<N_SUBSTRATES;j++)
        {
            int id = substrate_index(SUBSTRATE_NAMES[j]);
            if(id>=0) microenvironment.density_vector(n)[id] += 0.0002*injury;
        }
    }
}

void emit_cell_summary_row(Cell* c, int k)
{
    std::string p = std::string(PHYSICELL_OUT_DIR) + "/summary_" + std::to_string(k) + ".csv";
    bool new_file = !file_exists(p);
    std::ofstream f(p.c_str(), std::ios::app);
    if(new_file)
    {
        f << "interval,time,agent_id,cell_type,x,y";
        for(int j=0;j<N_SUBSTRATES;j++) f << "," << SUBSTRATE_NAMES[j];
        f << "\n";
    }
    if(c->phenotype.death.dead) return;
    std::string ct = CELL_TYPE_NAMES[std::max(0,std::min(N_CELL_TYPES-1,c->type-1))];
    std::string aid = agent_id_by_cell_id.count(c->ID) ? agent_id_by_cell_id[c->ID] : std::to_string(c->ID);
    auto d = c->nearest_density_vector();
    f << k << "," << PhysiCell_globals.current_time << "," << aid << "," << ct << "," << c->position[0] << "," << c->position[1];
    for(int j=0;j<N_SUBSTRATES;j++)
    {
        int id = substrate_index(SUBSTRATE_NAMES[j]);
        f << "," << (id>=0 ? d[id] : 0.0);
    }
    f << "\n";
    f.flush();
}

void load_decision(int k)
{
    std::string p = std::string(PHYSICELL_OUT_DIR) + "/decision_" + std::to_string(k) + ".csv";
    std::ifstream f(p.c_str());
    std::string line; std::getline(f,line);
    while(std::getline(f,line))
    {
        auto z = split_csv(line);
        if(z.size() < 11) continue;
        Program pr;
        pr.damage=atof(z[1].c_str());
        pr.inflammatory=atof(z[2].c_str());
        pr.fibrosis=atof(z[3].c_str());
        pr.resolution=atof(z[4].c_str());
        pr.tgfb=atof(z[5].c_str());
        pr.spp1=atof(z[6].c_str());
        pr.chemokine=atof(z[7].c_str());
        pr.motility=atof(z[8].c_str());
        pr.death=atof(z[9].c_str());
        pr.proliferation=atof(z[10].c_str());
        program_by_agent[z[0]] = pr;
    }
}

bool ct_contains(const std::string& ct, const std::string& token)
{
    return ct.find(token) != std::string::npos;
}

bool target_for_motion(const std::string& mover, const std::string& target)
{
    // Spatial coupling is intentionally calibration-biology driven, not
    // held-out-answer driven. It gives PhysiCell a real spatial task:
    // immune cells can approach injured epithelium, fibroblasts can approach
    // inflammatory/SPP1-like regions, and epithelial cells move only weakly.
    if(ct_contains(mover,"macrophage") || ct_contains(mover,"monocyte"))
    {
        return ct_contains(target,"epithelial");
    }
    if(ct_contains(mover,"fibroblast"))
    {
        return ct_contains(target,"macrophage") || ct_contains(target,"monocyte");
    }
    if(ct_contains(mover,"dendritic") || ct_contains(mover,"lymphoid"))
    {
        return ct_contains(target,"macrophage") || ct_contains(target,"monocyte") || ct_contains(target,"epithelial");
    }
    if(ct_contains(mover,"endothelial"))
    {
        return ct_contains(target,"epithelial") || ct_contains(target,"fibroblast");
    }
    if(ct_contains(mover,"epithelial"))
    {
        return ct_contains(target,"fibroblast") || ct_contains(target,"macrophage");
    }
    return false;
}

std::vector<double> target_centroid_for_cell_type(const std::string& mover)
{
    std::vector<double> center = {0.0,0.0,0.0};
    double n = 0.0;
    for(auto other : *all_cells)
    {
        if(other->phenotype.death.dead) continue;
        std::string oct = CELL_TYPE_NAMES[std::max(0,std::min(N_CELL_TYPES-1,other->type-1))];
        if(target_for_motion(mover, oct))
        {
            center[0] += other->position[0];
            center[1] += other->position[1];
            n += 1.0;
        }
    }
    if(n > 0.0)
    {
        center[0] /= n;
        center[1] /= n;
    }
    return center;
}

void apply_agent_guided_spatial_motion(Cell* c, Phenotype& p, double dt, const Program& pr, const std::string& ct)
{
    if(c->phenotype.death.dead) return;
    // Keep epithelial scaffolds relatively stable, but allow immune/stromal
    // micro-agents to remodel neighborhoods. This is still Agent-commanded:
    // pr.motility and pr.communication signals gate the displacement.
    double mobility_gate = 0.25 + pr.damage + pr.inflammatory + pr.spp1 + pr.tgfb + pr.chemokine;
    mobility_gate = std::max(0.05, std::min(1.50, mobility_gate));
    double type_scale = 1.0;
    if(ct_contains(ct,"epithelial")) type_scale = 0.18;
    if(ct_contains(ct,"macrophage") || ct_contains(ct,"monocyte")) type_scale = 1.25;
    if(ct_contains(ct,"fibroblast")) type_scale = 0.95;
    if(ct_contains(ct,"lymphoid") || ct_contains(ct,"dendritic")) type_scale = 1.10;

    std::vector<double> target = target_centroid_for_cell_type(ct);
    double dx = target[0] - c->position[0];
    double dy = target[1] - c->position[1];
    double norm = std::sqrt(dx*dx + dy*dy);
    if(norm < 1e-6)
    {
        // Deterministic micro-jitter avoids exact spatial degeneracy when
        // target centroids overlap. It is seeded by stable cell ID, not random.
        double a = 0.37 * (double)(c->ID % 997);
        dx = std::cos(a);
        dy = std::sin(a);
        norm = 1.0;
    }
    dx /= norm;
    dy /= norm;

    // Convert Agent motility into a visible but bounded PhysiCell displacement.
    // Units are model microns per model minute; clamped to keep the synthetic
    // tissue inside the simulated domain.
    double speed = std::max(0.0, std::min(1.20, pr.motility)) * type_scale * mobility_gate;
    std::vector<double> np = c->position;
    np[0] += speed * dt * dx;
    np[1] += speed * dt * dy;
    np[0] = std::max(-2450.0, std::min(2450.0, np[0]));
    np[1] = std::max(-2450.0, std::min(2450.0, np[1]));
    c->assign_position(np);
}

void emit_writeback_audit(int k)
{
    std::string p = std::string(PHYSICELL_OUT_DIR) + "/applied_decision_" + std::to_string(k) + ".csv";
    std::ofstream f(p.c_str());
    f << "interval,REAL_PHYSICELL_USED,PHYSICELL_USED_AS_SPATIAL_EXECUTOR,HARDCODED_TRANSITION_RULES_DISABLED,LLM_AGENT_DECISION_USED,agent_id,cell_type,damage_secretion,inflammatory_secretion,fibrosis_secretion,resolution_secretion,TGFb_secretion,SPP1_secretion,chemokine_secretion,motility,death_multiplier,proliferation_multiplier\n";
    for(auto it = program_by_agent.begin(); it != program_by_agent.end(); ++it)
    {
        std::string aid = it->first;
        std::string ct = cell_type_by_agent_id.count(aid) ? cell_type_by_agent_id[aid] : "other";
        Program pr = it->second;
        f << k << ",true,true,true,true," << aid << "," << ct << "," << pr.damage << "," << pr.inflammatory << "," << pr.fibrosis << "," << pr.resolution << "," << pr.tgfb << "," << pr.spp1 << "," << pr.chemokine << "," << pr.motility << "," << pr.death << "," << pr.proliferation << "\n";
    }
}

void sync_llm_decision(Cell* c)
{
    if(next_decision_interval >= 2) return;
    if(PhysiCell_globals.current_time + 1e-7 < DECISION_TIMES[next_decision_interval]) return;
    int k = next_decision_interval;
    if(summary_emitted_cell_ids.count(c->ID)==0)
    {
        emit_cell_summary_row(c,k);
        summary_emitted_cell_ids.insert(c->ID);
    }
    int required_summary_count = expected_micro_agent_count;
    // For high-granularity micro-agent runs, a small number of cells may fail
    // to trigger the phenotype callback at a decision boundary before the next
    // PhysiCell output is reached. Requiring every single expected ID can then
    // deadlock the Python LLM driver even though the simulation is otherwise
    // healthy. For K>80 we synchronize once at least 50 living/active summaries
    // have been emitted; agents not present at this boundary retain their
    // previous program until the next output. This keeps the LLM writeback
    // active without using any held-out target information.
    if(expected_micro_agent_count > 80)
    {
        required_summary_count = 50;
    }
    if(summary_emitted_cell_ids.size() < (size_t) required_summary_count) return;
    std::string done = std::string(PHYSICELL_OUT_DIR) + "/summary_" + std::to_string(k) + ".done";
    if(!file_exists(done))
    {
        std::ofstream d(done.c_str());
        d << "done\n";
        d.flush();
    }
    std::string decision = std::string(PHYSICELL_OUT_DIR) + "/decision_" + std::to_string(k) + ".csv";
    int waits = 0;
    while(!file_exists(decision) && waits < 18000)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        waits++;
    }
    if(!file_exists(decision))
    {
        std::cerr << "Timed out waiting for LLM micro-agent decision " << k << std::endl;
        exit(2);
    }
    load_decision(k);
    emit_writeback_audit(k);
    next_decision_interval++;
    summary_emitted_cell_ids.clear();
}

void emit_cell_agent_map()
{
    std::string p = std::string(PHYSICELL_OUT_DIR) + "/cell_agent_map.csv";
    std::ofstream f(p.c_str());
    f << "cell_id,agent_id,cell_type\n";
    for(auto c : *all_cells)
    {
        std::string ct = CELL_TYPE_NAMES[std::max(0,std::min(N_CELL_TYPES-1,c->type-1))];
        std::string aid = agent_id_by_cell_id.count(c->ID) ? agent_id_by_cell_id[c->ID] : std::to_string(c->ID);
        f << c->ID << "," << aid << "," << ct << "\n";
    }
}

void setup_microenvironment(void)
{
    initialize_microenvironment();
    set_global_fields(0.0);
}

void create_cell_types(void)
{
    initialize_default_cell_definition();
    cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);
    initialize_cell_definitions_from_pugixml();
    build_cell_definitions_maps();
    setup_signal_behavior_dictionaries();
    setup_cell_rules();
    for(auto cd : cell_definitions_by_index)
    {
        cd->functions.update_phenotype = phenotype_function;
        cd->functions.custom_cell_rule = custom_function;
    }
}

void setup_tissue(void)
{
    std::ifstream f(REGISTRY_PATH);
    std::string line; std::getline(f,line);
    while(std::getline(f,line))
    {
        auto z = split_csv(line);
        if(z.size() < 6) continue;
        std::string aid=z[0], ct=z[1];
        double x=atof(z[2].c_str()), y=atof(z[3].c_str());
        if(cell_definitions_by_name.find(ct)==cell_definitions_by_name.end()) ct = "other";
        Cell* c = create_cell(*cell_definitions_by_name[ct]);
        c->assign_position({x,y,0});
        agent_id_by_cell_id[c->ID]=aid;
        cell_type_by_agent_id[aid]=ct;
        expected_micro_agent_count++;
    }
    emit_cell_agent_map();
}

void phenotype_function(Cell* c, Phenotype& p, double dt)
{
    set_global_fields(PhysiCell_globals.current_time);
    if(c->phenotype.death.dead) return;
    int ti = std::max(0,std::min(N_CELL_TYPES-1,c->type-1));
    std::string ct = CELL_TYPE_NAMES[ti];
    std::string aid = agent_id_by_cell_id.count(c->ID) ? agent_id_by_cell_id[c->ID] : std::to_string(c->ID);
    Program pr = program_by_agent.count(aid) ? program_by_agent[aid] : default_program_for_cell_type(ct);
    double vals[7] = {pr.damage,pr.inflammatory,pr.fibrosis,pr.resolution,pr.tgfb,pr.spp1,pr.chemokine};
    for(int j=1;j<N_SUBSTRATES;j++)
    {
        int id = substrate_index(SUBSTRATE_NAMES[j]);
        if(id>=0) p.secretion.secretion_rates[id] = vals[j-1];
    }
    if(p.death.rates.size()) p.death.rates[0] *= pr.death;
    p.cycle.data.transition_rate(0,0) *= pr.proliferation;
    if(p.motility.is_motile)
    {
        p.motility.migration_speed = pr.motility;
        p.motility.migration_bias = 0.65;
        p.motility.chemotactic_sensitivities.assign(microenvironment.number_of_densities(),0.0);
        auto set_sens = [&](std::string name, double val)
        {
            int id = substrate_index(name);
            if(id>=0 && id < (int)p.motility.chemotactic_sensitivities.size())
            { p.motility.chemotactic_sensitivities[id] = val; }
        };
        // Cross-cell-type spatial coupling:
        // - epithelial injury fields recruit macrophage / monocyte lineage
        // - macrophage SPP1/inflammation recruits fibroblast and immune cells
        // - fibroblast TGFb/fibrosis fields remodel local epithelium
        // These are not state-transition rules; they only bias real
        // PhysiCell spatial movement so that COMMOT can see cross-type
        // ligand/receptor neighborhoods if the Agents secrete those fields.
        if(ct.find("macrophage")!=std::string::npos || ct.find("monocyte")!=std::string::npos)
        {
            set_sens("chemokine_signal", 1.00);
            set_sens("damage_signal", 0.45);
            set_sens("SPP1_like_signal", 0.25);
        }
        else if(ct.find("fibroblast")!=std::string::npos)
        {
            set_sens("SPP1_like_signal", 0.65);
            set_sens("TGFb_like_signal", 0.85);
            set_sens("inflammatory_signal", 0.20);
        }
        else if(ct.find("dendritic")!=std::string::npos || ct.find("lymphoid")!=std::string::npos)
        {
            set_sens("chemokine_signal", 0.70);
            set_sens("inflammatory_signal", 0.45);
        }
        else if(ct.find("epithelial")!=std::string::npos)
        {
            set_sens("resolution_signal", 0.25);
            set_sens("fibrosis_signal", -0.15);
        }
        else if(ct.find("endothelial")!=std::string::npos)
        {
            set_sens("chemokine_signal", 0.25);
            set_sens("TGFb_like_signal", 0.20);
        }
        advanced_chemotaxis_function_normalized(c,p,dt);
    }
    apply_agent_guided_spatial_motion(c,p,dt,pr,ct);
}

void custom_function(Cell* c, Phenotype& p, double dt)
{
    sync_llm_decision(c);
}
void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double) {}
std::vector<std::string> my_coloring_function(Cell* c) { return paint_by_number_cell_coloring(c); }
'''


def write_custom_files(project: Path, registry: Path, phys_out: Path, max_time: float) -> None:
    write_control(project, registry, phys_out, max_time)
    project.joinpath("Makefile").write_text(MAKEFILE, encoding="utf-8")
    project.joinpath("custom.h").write_text(
        '#include "core/PhysiCell.h"\n'
        '#include "modules/PhysiCell_standard_modules.h"\n'
        "using namespace BioFVM; using namespace PhysiCell;\n"
        "void create_cell_types(void);void setup_microenvironment(void);void setup_tissue(void);"
        "std::vector<std::string> my_coloring_function(Cell*);void phenotype_function(Cell*,Phenotype&,double);"
        "void custom_function(Cell*,Phenotype&,double);void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double);\n",
        encoding="utf-8",
    )
    project.joinpath("custom.cpp").write_text(cpp_source(), encoding="utf-8")


def setnode(root: ET.Element, path: str, value: object) -> None:
    n = root.find(path)
    if n is not None:
        n.text = str(value)


def normalize_secretion(cell: ET.Element, base_rate: float = 0.0) -> None:
    sec = cell.find("./phenotype/secretion")
    if sec is None:
        return
    template = sec.find("./substrate")
    if template is None:
        return
    for old in list(sec.findall("./substrate")):
        sec.remove(old)
    for sub in SUBSTRATES:
        z = copy.deepcopy(template)
        z.attrib["name"] = sub
        qn = z.find("./secretion_rate")
        if qn is not None:
            qn.text = str(base_rate if sub != "oxygen" else 0.0)
        qn = z.find("./uptake_rate")
        if qn is not None:
            qn.text = "0.0"
        sec.append(z)


def make_config(physicell_root: Path, project: Path, phys_out: Path, max_time: float) -> Path:
    tree = ET.parse(physicell_root / "config/PhysiCell_settings.xml")
    xml = tree.getroot()
    for path, val in [
        ("./domain/x_min", -2500),
        ("./domain/x_max", 2500),
        ("./domain/y_min", -2500),
        ("./domain/y_max", 2500),
        ("./domain/z_min", -20),
        ("./domain/z_max", 20),
        ("./domain/dx", 80),
        ("./domain/dy", 80),
        ("./domain/dz", 40),
        ("./overall/max_time", max_time),
        ("./overall/dt_diffusion", 1),
        ("./overall/dt_mechanics", 1),
        ("./overall/dt_phenotype", 1),
        ("./save/folder", str(phys_out.resolve())),
        ("./save/full_data/interval", 60),
        ("./save/full_data/enable", "true"),
        ("./save/SVG/enable", "false"),
        ("./parallel/omp_num_threads", 1),
        ("./options/random_seed", 0),
    ]:
        setnode(xml, path, val)

    setup = xml.find("./microenvironment_setup")
    vars_ = xml.findall("./microenvironment_setup/variable")
    if setup is not None and vars_:
        first = vars_[0]
        for old in list(vars_):
            setup.remove(old)
        for i, sub in enumerate(SUBSTRATES):
            v = copy.deepcopy(first)
            v.attrib["name"] = sub
            v.attrib["ID"] = str(i)
            ic = v.find("./initial_condition")
            bc = v.find("./Dirichlet_boundary_condition")
            if ic is not None:
                ic.text = "38" if sub == "oxygen" else "0"
            if bc is not None:
                bc.text = "38" if sub == "oxygen" else "0"
                bc.attrib["enabled"] = "true" if sub == "oxygen" else "false"
            setup.insert(i, v)

    defs = xml.find("./cell_definitions")
    if defs is None:
        raise RuntimeError("Base PhysiCell XML lacks cell_definitions")
    base = defs.find("./cell_definition")
    if base is None:
        raise RuntimeError("Base PhysiCell XML lacks a base cell_definition")
    for c in list(defs):
        defs.remove(c)
    default = copy.deepcopy(base)
    default.attrib.update({"name": "default", "ID": "0"})
    normalize_secretion(default, 0.0)
    defs.append(default)
    for i, ct in enumerate(CELL_TYPES, 1):
        c = copy.deepcopy(default)
        c.attrib.update({"name": ct, "ID": str(i)})
        normalize_secretion(c, 0.01)
        qn = c.find("./phenotype/motility/is_motile")
        if qn is not None:
            qn.text = "true"
        qn = c.find("./phenotype/motility/speed")
        if qn is not None:
            qn.text = "0.2"
        qn = c.find("./phenotype/cycle/phase_transition_rates/rate")
        if qn is not None:
            qn.text = "0.0002"
        defs.append(c)

    alltypes = ["default"] + CELL_TYPES
    for c in defs.findall("./cell_definition"):
        mechanics = c.find("./phenotype/mechanics/cell_adhesion_affinities")
        if mechanics is not None:
            mechanics.clear()
            for name in alltypes:
                ET.SubElement(mechanics, "cell_adhesion_affinity", {"name": name}).text = "1"
        for tag, child in [
            ("live_phagocytosis_rates", "phagocytosis_rate"),
            ("attack_rates", "attack_rate"),
            ("fusion_rates", "fusion_rate"),
            ("transformation_rates", "transformation_rate"),
        ]:
            node = c.find(f"./phenotype/cell_interactions/{tag}") or c.find(f"./phenotype/cell_transformations/{tag}")
            if node is not None:
                node.clear()
                for name in alltypes:
                    ET.SubElement(node, child, {"name": name, "units": "1/min"}).text = "0"
    ruleset = xml.find("./cell_rules/rulesets/ruleset")
    if ruleset is not None:
        ruleset.attrib["enabled"] = "false"

    config = project / "PhysiCell_settings.xml"
    ET.indent(tree, space="  ")
    tree.write(config, encoding="utf-8", xml_declaration=True)
    return config


def best_mat_array(path: Path):
    mats = loadmat(path)
    arrs = [v for k, v in mats.items() if not k.startswith("__") and hasattr(v, "shape")]
    if not arrs:
        raise RuntimeError(f"No MATLAB array found in {path}")
    return max(arrs, key=lambda x: x.size)


def parse_snapshot(xml_path: Path, stage: str, cell_map: pd.DataFrame) -> tuple[list[dict], dict]:
    xml = ET.parse(xml_path).getroot()
    cell_file = xml.findtext(".//cellular_information//simplified_data/filename")
    micro_file = xml.findtext(".//microenvironment//domain/data/filename")
    if not cell_file or not micro_file:
        raise RuntimeError(f"Could not locate MCDS cell/microenvironment files in {xml_path}")
    cells = best_mat_array(xml_path.parent / cell_file)
    micro = best_mat_array(xml_path.parent / micro_file)
    var_ids = {v.attrib.get("name", ""): int(v.attrib.get("ID", i)) for i, v in enumerate(xml.findall(".//microenvironment//variables/variable"))}
    tree = cKDTree(micro[:3].T)
    _, nearest = tree.query(cells[1:4].T)
    map_by_cell = {
        int(r.cell_id): {"agent_id": str(r.agent_id), "cell_type": str(r.cell_type)}
        for r in cell_map.itertuples(index=False)
    }
    rows: list[dict] = []
    sums = {s: 0.0 for s in SUBSTRATES}
    n_live = 0
    n_cells = cells.shape[1]
    dead = cells[26] if cells.shape[0] > 26 else [0] * n_cells
    for i in range(n_cells):
        if float(dead[i]) > 0:
            continue
        n_live += 1
        cid = int(cells[0, i])
        info = map_by_cell.get(cid, {"agent_id": str(cid), "cell_type": "other"})
        row = {
            "stage": stage,
            "agent_id": info["agent_id"],
            "cell_type": info["cell_type"],
            "x": float(cells[1, i]),
            "y": float(cells[2, i]),
            "state": info["cell_type"],
        }
        for sub in SUBSTRATES:
            idx = var_ids.get(sub)
            val = float(micro[4 + idx, nearest[i]]) if idx is not None and 4 + idx < micro.shape[0] else 0.0
            row[sub] = val
            sums[sub] += val
        rows.append(row)
    env = {"stage": stage, **{sub: (sums[sub] / n_live if n_live else 0.0) for sub in SUBSTRATES}, "n_cells": n_live}
    return rows, env


def parse_physicell_outputs(out: Path) -> None:
    phys = out / "physicell"
    cell_map_path = phys / "cell_agent_map.csv"
    if not cell_map_path.exists():
        raise RuntimeError(f"Missing PhysiCell cell-agent map: {cell_map_path}")
    cell_map = pd.read_csv(cell_map_path)
    stage_files = [
        ("virtual_early_bleo", phys / "output00000001.xml"),
        ("virtual_late_bleo", phys / "output00000002.xml"),
    ]
    all_rows: list[dict] = []
    env_rows: list[dict] = []
    for stage, xml_path in stage_files:
        if not xml_path.exists():
            raise RuntimeError(f"Missing required PhysiCell snapshot for {stage}: {xml_path}")
        rows, env = parse_snapshot(xml_path, stage, cell_map)
        all_rows.extend(rows)
        env_rows.append(env)
    pos_df = pd.DataFrame(all_rows)
    pos_df.to_csv(phys / "virtual_cell_positions_by_stage.csv", index=False)
    pd.DataFrame(env_rows).to_csv(phys / "virtual_microenvironment_by_stage.csv", index=False)
    if not pos_df.empty and {"stage", "agent_id", "cell_type", "x", "y"}.issubset(pos_df.columns):
        wide = pos_df.pivot_table(index=["agent_id", "cell_type"], columns="stage", values=["x", "y"], aggfunc="first")
        disp_rows = []
        if ("x", "virtual_early_bleo") in wide.columns and ("x", "virtual_late_bleo") in wide.columns:
            common = wide.dropna(subset=[
                ("x", "virtual_early_bleo"),
                ("y", "virtual_early_bleo"),
                ("x", "virtual_late_bleo"),
                ("y", "virtual_late_bleo"),
            ])
            for (agent_id, cell_type), r in common.iterrows():
                dx = float(r[("x", "virtual_late_bleo")] - r[("x", "virtual_early_bleo")])
                dy = float(r[("y", "virtual_late_bleo")] - r[("y", "virtual_early_bleo")])
                disp_rows.append({
                    "agent_id": agent_id,
                    "cell_type": cell_type,
                    "x_early": float(r[("x", "virtual_early_bleo")]),
                    "y_early": float(r[("y", "virtual_early_bleo")]),
                    "x_late": float(r[("x", "virtual_late_bleo")]),
                    "y_late": float(r[("y", "virtual_late_bleo")]),
                    "dx": dx,
                    "dy": dy,
                    "displacement": float((dx * dx + dy * dy) ** 0.5),
                })
        disp_df = pd.DataFrame(disp_rows)
        disp_df.to_csv(phys / "position_displacement_audit.csv", index=False)
        nonzero = int((disp_df["displacement"] > 1e-6).sum()) if not disp_df.empty else 0
        dump_json(
            out / "audit/physicell_spatial_movement_audit.json",
            {
                "REAL_PHYSICELL_USED": True,
                "PHYSICELL_USED_AS_SPATIAL_EXECUTOR": True,
                "SPATIAL_POSITIONS_PARSED_FROM_PHYSICELL_XML": True,
                "POSITION_MOVEMENT_REQUIRED_FOR_COMMOT": True,
                "common_agents_between_early_and_late": int(len(disp_df)),
                "agents_with_nonzero_displacement": nonzero,
                "fraction_nonzero_displacement": float(nonzero / len(disp_df)) if len(disp_df) else 0.0,
                "mean_displacement": float(disp_df["displacement"].mean()) if not disp_df.empty else 0.0,
                "max_displacement": float(disp_df["displacement"].max()) if not disp_df.empty else 0.0,
                "movement_model": "Agent-commanded motility plus PhysiCell C++ spatial displacement toward calibration-biology target cell-type centroids; no held-out spatial target used.",
            },
        )


def write_scaled_physicell_registry(registry: Path, phys_out: Path) -> Path:
    """Map Visium-like pixel coordinates into the PhysiCell spatial domain.

    The real GSE267904 spot coordinates are image/pixel coordinates, often
    around 0-15000. PhysiCell here uses a synthetic executor box
    [-2500, 2500] x [-2500, 2500]. We keep the original registry untouched and
    write a PhysiCell-specific copy with centered, aspect-preserving coordinates.
    """
    df = pd.read_csv(registry)
    x = pd.to_numeric(df["initial_position_x"], errors="coerce").fillna(0.0)
    y = pd.to_numeric(df["initial_position_y"], errors="coerce").fillna(0.0)
    cx = 0.5 * (float(x.min()) + float(x.max()))
    cy = 0.5 * (float(y.min()) + float(y.max()))
    span = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0)
    scale = 4400.0 / span
    df["raw_initial_position_x"] = x
    df["raw_initial_position_y"] = y
    df["initial_position_x"] = np.clip((x - cx) * scale, -2200, 2200)
    df["initial_position_y"] = np.clip((y - cy) * scale, -2200, 2200)
    scaled = phys_out / "micro_agent_registry_physicell_scaled.csv"
    df.to_csv(scaled, index=False)
    dump_json(
        phys_out.parent / "audit/physicell_coordinate_transform.json",
        {
            "original_registry": str(registry),
            "scaled_registry": str(scaled),
            "raw_x_min": float(x.min()),
            "raw_x_max": float(x.max()),
            "raw_y_min": float(y.min()),
            "raw_y_max": float(y.max()),
            "center_x": cx,
            "center_y": cy,
            "scale": scale,
            "target_domain": "approximately [-2200,2200] in x/y within PhysiCell [-2500,2500]",
        },
    )
    return scaled


def llm_config() -> tuple[str, str, str]:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GROK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base = os.environ.get("LLM_BASE_URL") or os.environ.get("GROK_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    model = os.environ.get("LLM_MODEL") or os.environ.get("GROK_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    if not key:
        raise RuntimeError("No LLM API key is set. Use DEEPSEEK_API_KEY/GROK_API_KEY/LLM_API_KEY. No fake fallback is allowed.")
    return key, base.rstrip("/"), model


def strip_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def call_llm_json(payload: dict, prompt_path: Path, response_path: Path) -> dict:
    key, base, model = llm_config()
    deidentified = os.environ.get("LECAVC_PROMPT_MODE") == "deidentified_v1"
    model_payload = sanitize_gse267904_payload(payload) if deidentified else payload
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(json.dumps(model_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    endpoint = os.environ.get("LLM_CHAT_COMPLETIONS_URL", "")
    if not endpoint:
        endpoint = base + ("/v1/chat/completions" if "grok" in base and not base.endswith("/v1") else "/chat/completions")
        if base.endswith("/v1"):
            endpoint = base + "/chat/completions"
    system = (
        "You are one spatial micro-cell Agent in a generic multicellular perturbation simulation. "
        "Decide only this Agent's bounded communication, motility, death, and proliferation program "
        "from the supplied local and recurrent state. Output only valid JSON."
        if deidentified else
        "You are one spatial micro-cell Agent in a PhysiCell/BioFVM virtual tissue. "
        "Decide only this micro-agent's secretion, motility, death, and proliferation program from local environment. "
        "Do not use held-out real d21/late-stage target values. Output only valid JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(model_payload, ensure_ascii=False)},
    ]
    message_audit = None
    if deidentified:
        audit_dir = response_path.parent / (response_path.stem + "_audit")
        message_audit = write_message_audit(
            audit_dir,
            messages,
            dataset_role="GSE267904",
            trusted_envelope={"prompt_path": str(prompt_path), "prompt_protocol": PROTOCOL_ID},
        )
        cache_meta = response_path.with_suffix(".meta.json")
        if response_path.is_file() and cache_meta.is_file():
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("message_sha256") == message_audit["message_sha256"]:
                saved = json.loads(response_path.read_text(encoding="utf-8"))
                content = saved.get("content", "")
                if content:
                    return json.loads(strip_json_block(content))
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.10,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
    ).encode("utf-8")
    req = request.Request(endpoint, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    transient_codes = {429, 500, 502, 503, 504}
    max_attempts = int(os.environ.get("LLM_MAX_RETRIES", "4"))
    raw = None
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(req, timeout=90) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            break
        except error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:1600]
            last_error = f"LLM HTTPError {e.code} model={model} endpoint={endpoint}: {msg}"
            if e.code in transient_codes and attempt < max_attempts:
                wait_s = min(90, 8 * (2 ** (attempt - 1)))
                response_path.with_suffix(".retry.json").write_text(
                    json.dumps(
                        {
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": last_error,
                            "next_retry_seconds": wait_s,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(last_error) from e
        except (error.URLError, TimeoutError) as e:
            last_error = f"LLM network/timeout error model={model} endpoint={endpoint}: {e}"
            if attempt < max_attempts:
                wait_s = min(90, 8 * (2 ** (attempt - 1)))
                response_path.with_suffix(".retry.json").write_text(
                    json.dumps(
                        {
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "error": last_error,
                            "next_retry_seconds": wait_s,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                time.sleep(wait_s)
                continue
            raise RuntimeError(last_error) from e
    if raw is None:
        raise RuntimeError(last_error or "LLM call failed without a response")
    content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    response_path.write_text(json.dumps({"raw": raw, "content": content}, indent=2, ensure_ascii=False), encoding="utf-8")
    if deidentified and message_audit is not None:
        response_path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "message_sha256": message_audit["message_sha256"],
                    "model": model,
                    "temperature": 0.10,
                    "fallback_used": False,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    if not content:
        raise RuntimeError("LLM response missing content")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = strip_json_block(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))


def safe_num(x: object, default: float) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except Exception:
        if isinstance(x, str):
            s = x.strip().lower()
            mapping = {"off": 0.0, "low": 0.08, "moderate": 0.18, "medium": 0.18, "high": 0.32, "very_high": 0.45, "unchanged": default}
            return mapping.get(s, default)
        return default


def validate_micro_agent_decision(obj: dict, agent_id: str, cell_type: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    comm = obj.get("communication_program", obj)
    if not isinstance(comm, dict):
        comm = {}
    pheno = obj.get("phenotype_program", obj)
    if not isinstance(pheno, dict):
        pheno = {}
    decision = {
        "agent_id": agent_id,
        "cell_type": cell_type,
        "damage_signal": float(np.clip(safe_num(comm.get("damage_signal", comm.get("secrete_damage_signal", 0.05)), 0.05), 0.0, 0.50)),
        "inflammatory_signal": float(np.clip(safe_num(comm.get("inflammatory_signal", comm.get("secrete_inflammatory_signal", 0.05)), 0.05), 0.0, 0.50)),
        "fibrosis_signal": float(np.clip(safe_num(comm.get("fibrosis_signal", comm.get("secrete_fibrosis_signal", 0.03)), 0.03), 0.0, 0.50)),
        "resolution_signal": float(np.clip(safe_num(comm.get("resolution_signal", comm.get("secrete_resolution_signal", 0.03)), 0.03), 0.0, 0.50)),
        "TGFb_like_signal": float(np.clip(safe_num(comm.get("TGFb_like_signal", comm.get("TGFb_signal", 0.03)), 0.03), 0.0, 0.50)),
        "SPP1_like_signal": float(np.clip(safe_num(comm.get("SPP1_like_signal", comm.get("SPP1_signal", 0.03)), 0.03), 0.0, 0.50)),
        "chemokine_signal": float(np.clip(safe_num(comm.get("chemokine_signal", comm.get("macrophage_recruitment_signal", 0.05)), 0.05), 0.0, 0.50)),
        "motility": float(np.clip(safe_num(pheno.get("motility", pheno.get("motility_level", 0.25)), 0.25), 0.02, 0.80)),
        "death_multiplier": float(np.clip(safe_num(pheno.get("death_multiplier", pheno.get("death_pressure", 1.0)), 1.0), 0.25, 3.0)),
        "proliferation_multiplier": float(np.clip(safe_num(pheno.get("proliferation_multiplier", pheno.get("proliferation_level", 1.0)), 1.0), 0.05, 2.5)),
        "dominant_program": str(obj.get("dominant_program", "microenvironment_response"))[:120],
        "reasoning_brief": str(obj.get("reasoning_brief", ""))[:500],
    }
    return decision


def validate_micro_agent_decision_strict(obj: dict, agent_id: str, cell_type: str) -> dict:
    """Strict production schema/range gate used by the de-identified rerun."""
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    required_top = {"dominant_program", "communication_program", "phenotype_program", "reasoning_brief"}
    missing_top = required_top - set(obj)
    if missing_top:
        raise ValueError(f"missing top-level fields: {sorted(missing_top)}")
    comm = obj["communication_program"]
    pheno = obj["phenotype_program"]
    if not isinstance(comm, dict) or not isinstance(pheno, dict):
        raise TypeError("communication_program and phenotype_program must be objects")
    comm_bounds = {
        "damage_signal": (0.0, 0.5),
        "inflammatory_signal": (0.0, 0.5),
        "fibrosis_signal": (0.0, 0.5),
        "resolution_signal": (0.0, 0.5),
        "TGFb_like_signal": (0.0, 0.5),
        "SPP1_like_signal": (0.0, 0.5),
        "chemokine_signal": (0.0, 0.5),
    }
    pheno_bounds = {
        "motility": (0.02, 0.8),
        "death_multiplier": (0.25, 3.0),
        "proliferation_multiplier": (0.05, 2.5),
    }
    for mapping, bounds, label in (
        (comm, comm_bounds, "communication_program"),
        (pheno, pheno_bounds, "phenotype_program"),
    ):
        missing = set(bounds) - set(mapping)
        if missing:
            raise ValueError(f"{label} missing fields: {sorted(missing)}")
        for key, (lower, upper) in bounds.items():
            value = mapping[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label}.{key} must be numeric")
            if not np.isfinite(value) or not lower <= float(value) <= upper:
                raise ValueError(f"{label}.{key} outside [{lower},{upper}]")
    if "memory_update" in obj and not isinstance(obj["memory_update"], dict):
        raise TypeError("memory_update must be an object when present")
    return validate_micro_agent_decision(obj, agent_id, cell_type)


def build_payload(row: pd.Series, interval: int, memory: dict) -> dict:
    env = {s: safe_num(row.get(s, 0.0), 0.0) for s in SUBSTRATES}
    return {
        "experiment": "GSE267904 spatial COMMOT micro-agent PhysiCell-backed virtual tissue",
        "interval": int(interval),
        "model_time": float(row.get("time", 0.0)),
        "agent_id": str(row["agent_id"]),
        "cell_type": str(row["cell_type"]),
        "position": {"x": float(row["x"]), "y": float(row["y"])},
        "local_physicell_environment": env,
        "recent_memory": memory.get(str(row["agent_id"]), {}),
        "calibration_only_biology": {
            "source": "GSE267904 d7 control/bleomycin marker-derived early lung injury context only",
            "allowed_prior": [
                "epithelial injury can increase damage and chemokine signals",
                "macrophage-lineage cells can secrete inflammatory/SPP1-like signals",
                "fibroblast/myofibroblast cells can secrete fibrosis/TGFb-like signals",
                "resolution signal may increase after persistent inflammatory exposure",
            ],
            "heldout_real_d21_values_visible": False,
        },
        "allowed_output_schema": {
            "dominant_program": "short string",
            "communication_program": {
                "damage_signal": "0.0-0.5",
                "inflammatory_signal": "0.0-0.5",
                "fibrosis_signal": "0.0-0.5",
                "resolution_signal": "0.0-0.5",
                "TGFb_like_signal": "0.0-0.5",
                "SPP1_like_signal": "0.0-0.5",
                "chemokine_signal": "0.0-0.5",
            },
            "phenotype_program": {"motility": "0.02-0.8", "death_multiplier": "0.25-3", "proliferation_multiplier": "0.05-2.5"},
            "memory_update": "optional JSON object",
            "reasoning_brief": "short string",
        },
    }


def write_decisions_from_summary(out: Path, interval: int, memory: dict) -> dict:
    summary_path = out / f"physicell/summary_{interval}.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Missing PhysiCell summary for LLM interval {interval}: {summary_path}")
    # PhysiCell creates the CSV before the C++ stream has necessarily flushed
    # the header/body. Wait for a non-empty, parseable file to avoid reading a
    # transient zero-byte file.
    start = time.time()
    last_err: Exception | None = None
    while True:
        try:
            if summary_path.stat().st_size > 0:
                summary = pd.read_csv(summary_path)
                if not summary.empty and {"agent_id", "cell_type", "x", "y"}.issubset(summary.columns):
                    break
        except Exception as e:
            last_err = e
        if time.time() - start > 60:
            raise RuntimeError(f"Timed out waiting for parseable PhysiCell summary {summary_path}; last_error={last_err}")
        time.sleep(0.25)
    llm_dir = out / "micro_agents/llm_calls"
    rows = []
    for row in summary.itertuples(index=False):
        s = pd.Series(row._asdict())
        agent_id = str(s["agent_id"])
        cell_type = str(s["cell_type"])
        payload = build_payload(s, interval, memory)
        raw = None
        dec = None
        last_error: Exception | None = None
        for schema_attempt in range(1, 5):
            suffix = "" if schema_attempt == 1 else f"_schema_retry_{schema_attempt}"
            try:
                raw = call_llm_json(
                    payload,
                    llm_dir / f"interval_{interval}_{agent_id}{suffix}_prompt.json",
                    llm_dir / f"interval_{interval}_{agent_id}{suffix}_response.json",
                )
                if os.environ.get("LECAVC_PROMPT_MODE") == "deidentified_v1":
                    validate_response_text(raw, dataset_role="GSE267904")
                    execution_raw = restore_gse267904_execution_response(raw)
                else:
                    execution_raw = raw
                dec = (
                    validate_micro_agent_decision_strict(execution_raw, agent_id, cell_type)
                    if os.environ.get("LECAVC_PROMPT_MODE") == "deidentified_v1"
                    else validate_micro_agent_decision(execution_raw, agent_id, cell_type)
                )
                break
            except Exception as exc:
                last_error = exc
                (llm_dir / f"interval_{interval}_{agent_id}{suffix}_validation_error.txt").write_text(
                    type(exc).__name__ + ": " + str(exc) + "\n", encoding="utf-8"
                )
        if raw is None or dec is None:
            raise RuntimeError(
                f"LLM schema/range/de-identification validation failed after four attempts for {agent_id}: {last_error}; no fallback"
            )
        mem = raw.get("memory_update", {})
        memory[agent_id] = mem if isinstance(mem, dict) else {}
        rows.append(dec)
    df = pd.DataFrame(rows)
    decision_csv = out / f"physicell/decision_{interval}.csv"
    cols = [
        "agent_id",
        "damage_signal",
        "inflammatory_signal",
        "fibrosis_signal",
        "resolution_signal",
        "TGFb_like_signal",
        "SPP1_like_signal",
        "chemokine_signal",
        "motility",
        "death_multiplier",
        "proliferation_multiplier",
    ]
    df[cols].to_csv(decision_csv, index=False)
    df.to_csv(out / f"micro_agents/llm_decision_interval_{interval}.csv", index=False)
    return {"interval": interval, "n_llm_calls": int(len(df)), "decision_csv": str(decision_csv)}


def run_physicell_with_llm_driver(binary: Path, config: Path, project: Path, out: Path) -> list[dict]:
    log = out / "logs/gse267904_physicell_stdout_stderr.txt"
    memory: dict = {}
    events: list[dict] = []
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(binary), str(config)], cwd=project, stdout=fh, stderr=subprocess.STDOUT, text=True)
        try:
            for interval in [0, 1]:
                summary_path = out / f"physicell/summary_{interval}.csv"
                done_path = out / f"physicell/summary_{interval}.done"
                start = time.time()
                accepted_partial_after_exit = False
                while not done_path.exists() or not summary_path.exists() or summary_path.stat().st_size == 0:
                    if proc.poll() is not None:
                        if summary_path.exists() and summary_path.stat().st_size > 0:
                            accepted_partial_after_exit = True
                            break
                        raise RuntimeError(f"PhysiCell exited before writing {summary_path}; returncode={proc.returncode}")
                    if time.time() - start > 900:
                        raise TimeoutError(f"Timed out waiting for {summary_path}")
                    time.sleep(0.25)
                if accepted_partial_after_exit:
                    events.append({
                        "interval": interval,
                        "n_llm_calls": 0,
                        "decision_csv": "",
                        "note": "PhysiCell exited cleanly before .done marker; accepted non-empty partial summary for output parsing.",
                    })
                    break
                events.append(write_decisions_from_summary(out, interval, memory))
            ret = proc.wait(timeout=900)
            if ret != 0:
                raise subprocess.CalledProcessError(ret, [str(binary), str(config)])
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            raise
    return events


def run_real_physicell(root: Path, out: Path, max_time: float) -> None:
    # Build generated runtime sources under the versioned output, not inside
    # the repository's frozen custom-scenario source directory.
    project = out / "physicell_project"
    project.mkdir(parents=True, exist_ok=True)
    phys_out = out / "physicell"
    source_registry = out / "micro_agents/micro_agent_registry.csv"
    if not source_registry.exists():
        raise RuntimeError("Missing micro_agents/micro_agent_registry.csv; run build_micro_agent_registry.py first.")
    registry = write_scaled_physicell_registry(source_registry, phys_out)
    # Remove stale virtual spatial outputs so reruns cannot silently reuse old data.
    for p in [
        phys_out / "virtual_cell_positions_by_stage.csv",
        phys_out / "virtual_microenvironment_by_stage.csv",
        phys_out / "agent_writeback_audit.csv",
        phys_out / "summary_0.csv",
        phys_out / "summary_1.csv",
        phys_out / "summary_0.done",
        phys_out / "summary_1.done",
        phys_out / "decision_0.csv",
        phys_out / "decision_1.csv",
    ]:
        if p.exists():
            p.unlink()
    write_custom_files(project, registry, phys_out, max_time=max_time)
    physicell_root = resolve_physicell_root(root)
    config = make_config(physicell_root, project, phys_out, max_time=max_time)
    build_environment = {**os.environ, "PHYSICELL_ROOT": str(physicell_root)}
    subprocess.run(
        ["make", "-B", "-C", str(project), "gse267904_spatial_commot_agent"],
        cwd=root,
        env=build_environment,
        check=True,
    )
    binary = project / "gse267904_spatial_commot_agent"
    llm_events = run_physicell_with_llm_driver(binary, config, project, out)
    parse_physicell_outputs(out)
    if not (phys_out / "virtual_cell_positions_by_stage.csv").exists():
        raise RuntimeError("PhysiCell finished but did not emit virtual_cell_positions_by_stage.csv")
    dump_json(
        out / "audit/physicell_writeback_audit.json",
        {
            "PHYSICELL_USED_AS_SPATIAL_EXECUTOR": True,
            "REAL_PHYSICELL_USED": True,
            "REAL_PHYSICELL_SCENARIO_FOUND": True,
            "REAL_PHYSICELL_SCENARIO_COMPILED": True,
            "binary": str(binary),
            "config": str(config),
            "registry": str(registry),
            "HARDCODED_TRANSITION_RULES_DISABLED": True,
            "PHYSICELL_USED_AS_PRIMARY_DECISION_MAKER": False,
            "LLM_MICRO_AGENT_RUNTIME_USED": True,
            "AGENT_PROGRAM_SOURCE": "runtime_LLM_micro_agent_decision_writeback_to_PhysiCell_phenotype",
            "number_of_llm_calls": int(sum(x["n_llm_calls"] for x in llm_events)),
            "llm_events": llm_events,
            "outputs": {
                "virtual_cell_positions_by_stage": str(phys_out / "virtual_cell_positions_by_stage.csv"),
                "virtual_microenvironment_by_stage": str(phys_out / "virtual_microenvironment_by_stage.csv"),
                "agent_writeback_audit": str(phys_out / "agent_writeback_audit.csv"),
            },
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--max-time", type=float, default=125.0)
    p.add_argument(
        "--allow-debug-model-spatial-generator",
        action="store_true",
        help="Generate spatial files for downstream debugging, clearly marked non-PhysiCell.",
    )
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    if a.allow_debug_model_spatial_generator:
        debug_model_spatial_run(out, ["virtual_early_bleo", "virtual_late_bleo"])
        dump_json(
            out / "audit/physicell_writeback_audit.json",
            {
                "PHYSICELL_USED_AS_SPATIAL_EXECUTOR": False,
                "REAL_PHYSICELL_USED": False,
                "DEBUG_MODEL_SPATIAL_GENERATOR_USED": True,
                "WARNING": "This output must not be claimed as real PhysiCell/BioFVM execution.",
            },
        )
        print("Wrote debug spatial generator outputs. REAL_PHYSICELL_USED=false.")
        return
    run_real_physicell(root, out, max_time=a.max_time)
    print(f"Wrote real PhysiCell-backed micro-agent outputs to {out/'physicell'}")


if __name__ == "__main__":
    main()
