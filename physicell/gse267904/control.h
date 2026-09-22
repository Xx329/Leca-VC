#pragma once
#define REGISTRY_PATH "runtime/micro_agent_registry_physicell_scaled.csv"
#define PHYSICELL_OUT_DIR "runtime"
#define MAX_TIME 125.0
static const int N_CELL_TYPES=10;
static const int N_SUBSTRATES=8;
static const char* CELL_TYPE_NAMES[N_CELL_TYPES]={ "alveolar_epithelial_AT1_AT2","activated_Krt8_ADI_epithelial","airway_epithelial","macrophage","recruited_monocyte_macrophage","fibroblast_myofibroblast","endothelial","dendritic","lymphoid","other" };
static const char* SUBSTRATE_NAMES[N_SUBSTRATES]={ "oxygen","damage_signal","inflammatory_signal","fibrosis_signal","resolution_signal","TGFb_like_signal","SPP1_like_signal","chemokine_signal" };
