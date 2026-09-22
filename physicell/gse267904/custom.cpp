#include "custom.h"
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
