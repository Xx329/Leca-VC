#include "custom.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const int N_TYPES = 6;
static const char* TYPE_NAMES[N_TYPES] = {
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell"
};
static const double BASE_BIRTH[N_TYPES] = {0.0010, 0.0005, 0.0008, 0.0007, 0.0011, 0.0009};
static const double BASE_DEATH[N_TYPES] = {0.00030, 0.00035, 0.00040, 0.00035, 0.00030, 0.00032};

struct ExecutedParameters {
    double birth = 1.0, death = 1.0, motility = 1.0, secretion = 1.0, uptake = 1.0;
    double recruitment = 0.0, clearance = 0.0;
    double activation = 0.0, stress = 0.0, exhaustion = 0.0;
    double inflammatory = 0.0, suppression = 0.0;
    double intrinsic_budget = 0.0;
    double external_recruitment_budget = 0.0;
    double clearance_budget = 0.0;
};

static std::map<std::string, ExecutedParameters> policy;
static std::map<std::string, int> cumulative_baseline_divisions, cumulative_actuator_divisions;
static std::map<std::string, int> cumulative_baseline_apoptosis, cumulative_actuator_apoptosis;
static std::map<std::string, int> cumulative_actuator_recruited, cumulative_actuator_cleared;
static std::map<std::string, int> last_live;
static std::set<int> counted_dead_cells;
static std::set<int> known_cell_ids;
static std::map<std::string,double> intrinsic_residual, recruitment_residual, clearance_residual;
static std::map<std::string,int> last_baseline_divisions,last_actuator_divisions;
static std::map<std::string,int> last_baseline_apoptosis,last_actuator_apoptosis;
static std::map<std::string,int> last_actuator_recruited,last_actuator_cleared;
static std::set<int> completed_checkpoints;
static std::string run_dir, bridge_path, sample_id, model_name;
static int endpoint_emitted = 0;

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out; std::stringstream stream(line); std::string token;
    while(std::getline(stream, token, ',')) {if(!token.empty()&&token.back()=='\r')token.pop_back();out.push_back(token);}
    return out;
}

static int type_index(const std::string& name) {
    for(int i=0; i<N_TYPES; ++i) if(name == TYPE_NAMES[i]) return i;
    return -1;
}

static std::string cell_type(Cell* cell) {
    int index = cell->type - 1;
    return index >= 0 && index < N_TYPES ? TYPE_NAMES[index] : "INVALID";
}

static double substrate_mean(const std::string& name) {
    int index = microenvironment.find_density_index(name);
    if(index < 0 || microenvironment.number_of_voxels() == 0) return 0.0;
    double total = 0.0;
    for(int voxel=0; voxel<microenvironment.number_of_voxels(); ++voxel)
        total += microenvironment.density_vector(voxel)[index];
    return total / microenvironment.number_of_voxels();
}

static std::map<std::string,int> live_counts() {
    std::map<std::string,int> counts;
    for(int i=0;i<N_TYPES;++i) counts[TYPE_NAMES[i]]=0;
    for(auto cell : *all_cells) if(!cell->phenotype.death.dead) counts[cell_type(cell)]++;
    return counts;
}

static void refresh_event_counts() {
    for(auto cell : *all_cells) {
        if(!known_cell_ids.count(cell->ID)) {
            known_cell_ids.insert(cell->ID);
            cumulative_baseline_divisions[cell_type(cell)]++;
        }
        if(cell->phenotype.death.dead && !counted_dead_cells.count(cell->ID)) {
            counted_dead_cells.insert(cell->ID);
            cumulative_baseline_apoptosis[cell_type(cell)]++;
        }
    }
}

static void append_interval_events(int checkpoint,const std::map<std::string,int>& live) {
    if(checkpoint==0) {
        for(int i=0;i<N_TYPES;++i){std::string n=TYPE_NAMES[i];last_live[n]=live.at(n);last_baseline_divisions[n]=cumulative_baseline_divisions[n];last_actuator_divisions[n]=cumulative_actuator_divisions[n];last_baseline_apoptosis[n]=cumulative_baseline_apoptosis[n];last_actuator_apoptosis[n]=cumulative_actuator_apoptosis[n];last_actuator_recruited[n]=cumulative_actuator_recruited[n];last_actuator_cleared[n]=cumulative_actuator_cleared[n];}
        return;
    }
    std::string path=run_dir+"/v21_interval_event_audit.csv";bool fresh=!std::ifstream(path.c_str()).good();std::ofstream out(path.c_str(),std::ios::app);
    if(fresh)out<<"interval,cell_type,normalized_internal_minutes,initial_workers,baseline_divisions,actuator_divisions,baseline_apoptosis,actuator_apoptosis,actuator_recruitment,actuator_clearance,baseline_drift,actuator_net_effect,final_workers,conservation_error,intrinsic_residual,external_recruitment_residual,clearance_residual\n";
    for(int i=0;i<N_TYPES;++i){
        std::string n=TYPE_NAMES[i];int bd=cumulative_baseline_divisions[n]-last_baseline_divisions[n],ad=cumulative_actuator_divisions[n]-last_actuator_divisions[n],ba=cumulative_baseline_apoptosis[n]-last_baseline_apoptosis[n],aa=cumulative_actuator_apoptosis[n]-last_actuator_apoptosis[n],ar=cumulative_actuator_recruited[n]-last_actuator_recruited[n],ac=cumulative_actuator_cleared[n]-last_actuator_cleared[n];int predicted=last_live[n]+bd+ad+ar-ba-aa-ac;int error=live.at(n)-predicted;
        out<<checkpoint-1<<","<<n<<",2,"<<last_live[n]<<","<<bd<<","<<ad<<","<<ba<<","<<aa<<","<<ar<<","<<ac<<","<<bd-ba<<","<<ad+ar-aa-ac<<","<<live.at(n)<<","<<error<<","<<intrinsic_residual[n]<<","<<recruitment_residual[n]<<","<<clearance_residual[n]<<"\n";
        last_live[n]=live.at(n);last_baseline_divisions[n]=cumulative_baseline_divisions[n];last_actuator_divisions[n]=cumulative_actuator_divisions[n];last_baseline_apoptosis[n]=cumulative_baseline_apoptosis[n];last_actuator_apoptosis[n]=cumulative_actuator_apoptosis[n];last_actuator_recruited[n]=cumulative_actuator_recruited[n];last_actuator_cleared[n]=cumulative_actuator_cleared[n];
    }
}

static void append_trajectories(int checkpoint, const std::map<std::string,int>& live) {
    std::string composition_path=run_dir+"/composition_trajectory.csv";
    bool composition_new=!std::ifstream(composition_path.c_str()).good();
    std::ofstream composition(composition_path.c_str(),std::ios::app);
    if(composition_new) composition<<"checkpoint,normalized_percent,cell_type,live_cells,total_live,proportion\n";
    int total=0; for(auto const& item:live) total+=item.second;
    for(int i=0;i<N_TYPES;++i) {
        std::string name=TYPE_NAMES[i];
        composition<<checkpoint<<","<<checkpoint*20<<","<<name<<","<<live.at(name)<<","<<total<<","<<std::setprecision(12)<<(total?double(live.at(name))/total:0)<<"\n";
    }
    std::string count_path=run_dir+"/cell_count_trajectory.csv";
    bool count_new=!std::ifstream(count_path.c_str()).good();
    std::ofstream counts(count_path.c_str(),std::ios::app);
    if(count_new) counts<<"checkpoint,normalized_percent,total_live,total_dead,total_cells,cumulative_baseline_divisions,cumulative_actuator_divisions,cumulative_baseline_apoptosis,cumulative_actuator_apoptosis,cumulative_actuator_recruitment,cumulative_actuator_clearance\n";
    int dead=0,bd=0,ad=0,ba=0,aa=0,ar=0,ac=0;
    for(auto cell:*all_cells) dead+=cell->phenotype.death.dead;
    for(int i=0;i<N_TYPES;++i){std::string n=TYPE_NAMES[i];bd+=cumulative_baseline_divisions[n];ad+=cumulative_actuator_divisions[n];ba+=cumulative_baseline_apoptosis[n];aa+=cumulative_actuator_apoptosis[n];ar+=cumulative_actuator_recruited[n];ac+=cumulative_actuator_cleared[n];}
    counts<<checkpoint<<","<<checkpoint*20<<","<<total<<","<<dead<<","<<all_cells->size()<<","<<bd<<","<<ad<<","<<ba<<","<<aa<<","<<ar<<","<<ac<<"\n";
    std::string bio_path=run_dir+"/biofvm_trajectory.csv";
    bool bio_new=!std::ifstream(bio_path.c_str()).good();
    std::ofstream bio(bio_path.c_str(),std::ios::app);
    if(bio_new) bio<<"checkpoint,normalized_percent,treatment_signal,inflammatory_signal,suppressive_signal,survival_signal,stress_signal\n";
    bio<<checkpoint<<","<<checkpoint*20<<","<<substrate_mean("treatment_signal")<<","<<substrate_mean("inflammatory_signal")<<","<<substrate_mean("suppressive_signal")<<","<<substrate_mean("survival_signal")<<","<<substrate_mean("stress_signal")<<"\n";
}

static std::string emit_state(int checkpoint) {
    auto live=live_counts(); refresh_event_counts();append_interval_events(checkpoint,live); append_trajectories(checkpoint,live);
    std::string path=run_dir+"/state_checkpoint_"+std::to_string(checkpoint)+".csv";
    std::ofstream out(path.c_str());
    out<<"checkpoint,cell_type,live_cells,dead_cells,cumulative_baseline_divisions,cumulative_actuator_divisions,cumulative_baseline_apoptosis,cumulative_actuator_apoptosis,cumulative_actuator_recruitment,cumulative_actuator_clearance,mean_birth_rate,mean_death_rate,treatment_signal,inflammatory_signal,suppressive_signal,survival_signal,stress_signal\n";
    std::map<std::string,int> dead;for(int i=0;i<N_TYPES;++i)dead[TYPE_NAMES[i]]=0;
    for(auto cell:*all_cells)if(cell->phenotype.death.dead)dead[cell_type(cell)]++;
    for(int i=0;i<N_TYPES;++i){
        std::string name=TYPE_NAMES[i]; ExecutedParameters q=policy[name];
        out<<checkpoint<<","<<name<<","<<live[name]<<","<<dead[name]<<","<<cumulative_baseline_divisions[name]<<","<<cumulative_actuator_divisions[name]<<","<<cumulative_baseline_apoptosis[name]<<","<<cumulative_actuator_apoptosis[name]<<","<<cumulative_actuator_recruited[name]<<","<<cumulative_actuator_cleared[name]<<","<<BASE_BIRTH[i]<<","<<BASE_DEATH[i]<<","<<substrate_mean("treatment_signal")<<","<<substrate_mean("inflammatory_signal")<<","<<substrate_mean("suppressive_signal")<<","<<substrate_mean("survival_signal")<<","<<substrate_mean("stress_signal")<<"\n";
    }
    return path;
}

static int rounded_with_residual(double requested,double& residual) {
    double due=requested+residual;int scheduled=due>=0?int(std::floor(due+0.5)):int(std::ceil(due-0.5));residual=due-scheduled;return scheduled;
}

static void population_writeback(const std::string& name, ExecutedParameters& q,int checkpoint,int current_total) {
    std::vector<Cell*> candidates;
    for(auto cell:*all_cells) if(!cell->phenotype.death.dead && cell_type(cell)==name) candidates.push_back(cell);
    int current=candidates.size();
    double requested_intrinsic=q.intrinsic_budget*current;
    double requested_recruitment=q.external_recruitment_budget*current_total;
    double requested_clearance=q.clearance_budget*current;
    double intrinsic_before=intrinsic_residual[name],recruitment_before=recruitment_residual[name],clearance_before=clearance_residual[name];
    int intrinsic_scheduled=rounded_with_residual(requested_intrinsic,intrinsic_residual[name]);
    int recruitment_scheduled=rounded_with_residual(requested_recruitment,recruitment_residual[name]);
    int clearance_scheduled=rounded_with_residual(requested_clearance,clearance_residual[name]);
    int divide=std::max(0,intrinsic_scheduled),apoptosis=std::max(0,-intrinsic_scheduled);
    if(current==0&&divide>0){intrinsic_residual[name]+=divide;divide=0;}
    apoptosis=std::min(apoptosis,current);if(-intrinsic_scheduled>apoptosis)intrinsic_residual[name]-=(-intrinsic_scheduled-apoptosis);
    int remove=std::min(clearance_scheduled,current-apoptosis);if(clearance_scheduled>remove)clearance_residual[name]+=clearance_scheduled-remove;
    int add=std::max(0,recruitment_scheduled);
    for(int i=0;i<divide;++i){Cell* parent=candidates[i%candidates.size()];Cell* cell=create_cell(*cell_definitions_by_name[name]);cell->assign_position(parent->position);known_cell_ids.insert(cell->ID);cumulative_actuator_divisions[name]++;}
    for(int i=0;i<add;++i){
        double radius=900.0*std::sqrt(UniformRandom()),angle=6.283185307179586*UniformRandom();
        Cell* cell=create_cell(*cell_definitions_by_name[name]);
        cell->assign_position({radius*std::cos(angle),radius*std::sin(angle),0.0});
        known_cell_ids.insert(cell->ID);
        cumulative_actuator_recruited[name]++;
    }
    for(int i=0;i<apoptosis;++i){candidates[i]->start_death(0);counted_dead_cells.insert(candidates[i]->ID);cumulative_actuator_apoptosis[name]++;}
    for(int i=0;i<remove;++i){Cell* cell=candidates[apoptosis+i];cell->start_death(0);counted_dead_cells.insert(cell->ID);cumulative_actuator_cleared[name]++;}
    std::string path=run_dir+"/v21_budget_audit.csv";bool fresh=!std::ifstream(path.c_str()).good();std::ofstream out(path.c_str(),std::ios::app);
    if(fresh)out<<"checkpoint,cell_type,normalized_internal_minutes,current_type_workers,current_total_live_workers,intrinsic_population_budget,external_recruitment_budget,clearance_budget,requested_intrinsic_workers,requested_recruitment_workers,requested_clearance_workers,requested_increase_workers,requested_decrease_workers,intrinsic_residual_before,external_recruitment_residual_before,clearance_residual_before,scheduled_actuator_divisions,scheduled_actuator_apoptosis,scheduled_actuator_recruitment,scheduled_actuator_clearance,intrinsic_residual_after,external_recruitment_residual_after,clearance_residual_after\n";
    out<<checkpoint<<","<<name<<",2,"<<current<<","<<current_total<<","<<q.intrinsic_budget<<","<<q.external_recruitment_budget<<","<<q.clearance_budget<<","<<requested_intrinsic<<","<<requested_recruitment<<","<<requested_clearance<<","<<std::max(0.0,requested_intrinsic)+requested_recruitment<<","<<std::max(0.0,-requested_intrinsic)+requested_clearance<<","<<intrinsic_before<<","<<recruitment_before<<","<<clearance_before<<","<<divide<<","<<apoptosis<<","<<add<<","<<remove<<","<<intrinsic_residual[name]<<","<<recruitment_residual[name]<<","<<clearance_residual[name]<<"\n";
}

static void load_execution(int checkpoint, const std::string& state_path) {
    std::string path=run_dir+"/execution_parameters_"+std::to_string(checkpoint)+".csv";
    std::ifstream in(path.c_str()); if(!in.good()){std::cerr<<"Missing execution file "<<path<<"\n";exit(71);}
    std::string line;std::getline(in,line);auto header=split(line);std::map<std::string,int> column;
    for(int i=0;i<(int)header.size();++i)column[header[i]]=i;
    std::set<std::string> seen;std::map<std::string,ExecutedParameters> pending;
    while(std::getline(in,line)){
        auto row=split(line); if(row.size()<header.size())continue; std::string name=row[column["cell_type"]];
        if(type_index(name)<0||seen.count(name)){std::cerr<<"Invalid duplicate cell type action\n";exit(72);}seen.insert(name);
        ExecutedParameters q;
        q.birth=atof(row[column["birth_rate_multiplier"]].c_str());q.death=atof(row[column["death_rate_multiplier"]].c_str());
        q.motility=atof(row[column["motility_multiplier"]].c_str());q.secretion=atof(row[column["secretion_multiplier"]].c_str());q.uptake=atof(row[column["uptake_multiplier"]].c_str());
        q.recruitment=atof(row[column["recruitment_rate"]].c_str());q.clearance=atof(row[column["clearance_rate"]].c_str());q.activation=atof(row[column["activation_strength"]].c_str());q.stress=atof(row[column["stress_strength"]].c_str());q.exhaustion=atof(row[column["exhaustion_strength"]].c_str());q.inflammatory=atof(row[column["inflammatory_strength"]].c_str());q.suppression=atof(row[column["suppression_strength"]].c_str());
        q.intrinsic_budget=atof(row[column["intrinsic_population_budget"]].c_str());q.external_recruitment_budget=atof(row[column["external_recruitment_budget"]].c_str());q.clearance_budget=atof(row[column["clearance_budget"]].c_str());
        if(q.intrinsic_budget < -0.2000001 || q.intrinsic_budget > 0.2000001 || q.external_recruitment_budget < -1e-12 || q.external_recruitment_budget > 0.2000001 || q.clearance_budget < -1e-12 || q.clearance_budget > 0.2000001){std::cerr<<"V2.1 budget outside frozen bounds\n";exit(75);}pending[name]=q;
    }
    if(seen.size()!=N_TYPES){std::cerr<<"Execution file lacks six cell types\n";exit(73);}
    auto before=live_counts();int current_total=0;for(auto const& item:before)current_total+=item.second;
    for(int i=0;i<N_TYPES;++i){std::string n=TYPE_NAMES[i];auto q=pending[n];population_writeback(n,q,checkpoint,current_total);policy[n]=q;}
    std::string audit_path=run_dir+"/cpp_writeback_audit.csv";bool fresh=!std::ifstream(audit_path.c_str()).good();std::ofstream audit(audit_path.c_str(),std::ios::app);
    if(fresh)audit<<"checkpoint,cell_type,baseline_birth_rate,baseline_death_rate,motility_speed,secretion_rate,uptake_rate,intrinsic_population_budget,external_recruitment_budget,clearance_budget,writeback_executed\n";
    for(int i=0;i<N_TYPES;++i){std::string n=TYPE_NAMES[i];auto q=policy[n];audit<<checkpoint<<","<<n<<","<<BASE_BIRTH[i]<<","<<BASE_DEATH[i]<<","<<0.35*q.motility<<","<<0.01*q.secretion<<","<<0.002*q.uptake<<","<<q.intrinsic_budget<<","<<q.external_recruitment_budget<<","<<q.clearance_budget<<",true\n";}
}

static void online_checkpoint(int checkpoint) {
    if(completed_checkpoints.count(checkpoint))return;completed_checkpoints.insert(checkpoint);
    std::string state_path=emit_state(checkpoint);
    std::string execution=run_dir+"/execution_parameters_"+std::to_string(checkpoint)+".csv";std::remove(execution.c_str());
    std::string command="timeout 900s python \""+bridge_path+"\" --run-dir \""+run_dir+"\" --checkpoint "+std::to_string(checkpoint);
    int rc=std::system(command.c_str());if(rc!=0){std::cerr<<"Runtime bridge failed checkpoint "<<checkpoint<<" rc="<<rc<<"\n";exit(74);}
    load_execution(checkpoint,state_path);
}

void create_cell_types(void) {
    initialize_default_cell_definition();
    cell_defaults.phenotype.secretion.sync_to_microenvironment(&microenvironment);
    initialize_cell_definitions_from_pugixml();build_cell_definitions_maps();
    setup_signal_behavior_dictionaries();setup_cell_rules();
    for(auto definition:cell_definitions_by_index){definition->phenotype.secretion.sync_to_microenvironment(&microenvironment);definition->functions.update_phenotype=phenotype_function;definition->functions.custom_cell_rule=custom_function;}
}

void setup_microenvironment(void){initialize_microenvironment();}

void setup_tissue(void) {
    run_dir=parameters.strings("run_dir");bridge_path=parameters.strings("bridge_path");sample_id=parameters.strings("sample_id");model_name=parameters.strings("model_name");
    std::ifstream in(parameters.strings("registry_path").c_str());if(!in.good()){std::cerr<<"Missing worker registry\n";exit(61);}std::string line;std::getline(in,line);int count=0;
    while(std::getline(in,line)){auto row=split(line);if(row.size()<6)continue;std::string name=row[1];if(type_index(name)<0){std::cerr<<"Unknown registry cell type "<<name<<"\n";exit(62);}Cell* cell=create_cell(*cell_definitions_by_name[name]);cell->assign_position({atof(row[2].c_str()),atof(row[3].c_str()),atof(row[4].c_str())});count++;}
    if(count!=600){std::cerr<<"Expected 600 workers, got "<<count<<"\n";exit(63);}for(auto cell:*all_cells)known_cell_ids.insert(cell->ID);for(int i=0;i<N_TYPES;++i)policy[TYPE_NAMES[i]]=ExecutedParameters();
    online_checkpoint(0);
}

void phenotype_function(Cell* cell, Phenotype& phenotype, double dt) {
    int checkpoints[4]={1,2,3,4};double minutes[4]={2,4,6,8};
    for(int i=0;i<4;++i)if(PhysiCell_globals.current_time+1e-7>=minutes[i])online_checkpoint(checkpoints[i]);
    if(!endpoint_emitted&&PhysiCell_globals.current_time+1e-7>=10.0){endpoint_emitted=1;emit_state(5);}
    if(cell->phenotype.death.dead)return;std::string name=cell_type(cell);int index=type_index(name);ExecutedParameters q=policy[name];
    int treatment=microenvironment.find_density_index("treatment_signal"),inflammatory=microenvironment.find_density_index("inflammatory_signal"),suppressive=microenvironment.find_density_index("suppressive_signal"),survival=microenvironment.find_density_index("survival_signal"),stress=microenvironment.find_density_index("stress_signal");
    phenotype.cycle.data.transition_rate(0,0)=BASE_BIRTH[index];
    if(phenotype.death.rates.size())phenotype.death.rates[0]=BASE_DEATH[index];
    phenotype.motility.is_motile=true;phenotype.motility.migration_speed=0.35*q.motility;phenotype.motility.migration_bias=0.0;
    phenotype.secretion.uptake_rates[treatment]=0.002*q.uptake;
    phenotype.secretion.secretion_rates[inflammatory]=0.010*q.secretion*q.inflammatory;
    phenotype.secretion.secretion_rates[suppressive]=0.008*q.secretion*q.suppression;
    phenotype.secretion.secretion_rates[survival]=0.006*q.secretion*std::max(0.0,1.0-q.stress);
    phenotype.secretion.secretion_rates[stress]=0.008*q.secretion*(q.stress+0.5*q.exhaustion);
    for(int substrate:{inflammatory,suppressive,survival,stress})phenotype.secretion.saturation_densities[substrate]=1.0;
    cell->set_internal_uptake_constants(dt);
}

void custom_function(Cell*,Phenotype&,double){}
void contact_function(Cell*,Phenotype&,Cell*,Phenotype&,double){}
std::vector<std::string> my_coloring_function(Cell* cell){return paint_by_number_cell_coloring(cell);}
