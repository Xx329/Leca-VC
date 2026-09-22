#include "custom.h"
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
        if(mac){
        q.inflamm = clamp01(0.45 + 0.65*injury);  // 原来 0.30+0.55*injury
        q.macro = clamp01(0.40 + 0.70*inflam);    // 原来 0.25+0.65*inflam
        q.tgfb = clamp01(0.40 + 0.70*macro);      // 原来 0.25+0.55*macro
        }
        if(fib){
        q.tgfb = clamp01(0.45 + 0.65*tgfb);      // 原来 0.30+0.55*tgfb
        q.ecm = clamp01(0.40 + 0.70*(.5*tgfb+.5*macro)); // 原来 0.25+0.60
        }
    }
    double tgfb_block=condition_name=="TGFB_blockade"?1.0-intervention_dose:1.0;
    double macro_block=condition_name=="macrophage_SPP1_APOE_suppression"&&mac?1.0-intervention_dose:1.0;
    double repair_boost=condition_name=="epithelial_repair_promotion"&&epi?1.0+parameters.doubles("repair_intervention_multiplier")*intervention_dose:1.0;
    double ecm_block=condition_name=="fibroblast_ECM_suppression"&&fib?1.0-intervention_dose:1.0;
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
    if(mac){
        // 大幅上调炎症和 TGF-β 的分泌系数，取消 sink
        kernel_flux(c, ifn, 3.0 * q.inflamm * macro_activation * (0.9 + 0.6 * injury), 0.0, dt);
        kernel_flux(c, im, 2.0 * q.macro * macro_block * macro_activation * (1.2 + 1.0 * memory), 0, dt);
        kernel_flux(c, it, 2.0 * q.tgfb * tgfb_block * macro_block * macro_activation * (0.9 + 0.5 * macro + 0.5 * memory), 0, dt);
    }
    if(fib){
        double fibrotic_response = myofibroblast * (1.0 + 1.5 * memory);
        kernel_flux(c, it, 1.0 * q.tgfb * tgfb_block * fibrotic_response, 0, dt);
        kernel_flux(c, ie, 3.0 * q.ecm * ecm_block * fibrotic_response,
                    parameters.doubles("ecm_degradation_rate") * (1.0 + parameters.doubles("ecm_intervention_multiplier") * intervention_dose * (condition_name=="fibroblast_ECM_suppression")), dt);
    }
    if(contains(ct,"dendritic")||contains(ct,"lymphoid"))kernel_flux(c,ifn,.25*q.inflamm*(injury+inflam),0,dt);
    if(contains(ct,"endothelial"))kernel_flux(c,ih,.20*q.repair*(1.0-injury),0,dt);
    if(p.motility.is_motile){p.motility.migration_speed=.5*q.motility;p.motility.migration_bias=.35;p.motility.chemotactic_sensitivities.assign(microenvironment.number_of_densities(),0.0);p.motility.chemotactic_sensitivities[ii]=mac?.5:0.0;p.motility.chemotactic_sensitivities[it]=fib?.5:0.0;advanced_chemotaxis_function_normalized(c,p,dt);}
}
void custom_function(Cell* c,Phenotype& p,double dt){emit_due();}
void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double){}
std::vector<std::string> my_coloring_function(Cell* c){return paint_by_number_cell_coloring(c);}
std::vector<std::string> heterogeneity_coloring_function(Cell* pCell){return my_coloring_function(pCell);}
