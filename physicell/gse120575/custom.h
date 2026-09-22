#include "core/PhysiCell.h"
#include "modules/PhysiCell_standard_modules.h"

using namespace BioFVM;
using namespace PhysiCell;

void create_cell_types(void);
void setup_microenvironment(void);
void setup_tissue(void);
std::vector<std::string> my_coloring_function(Cell*);
void phenotype_function(Cell*, Phenotype&, double);
void custom_function(Cell*, Phenotype&, double);
void contact_function(Cell*, Phenotype&, Cell*, Phenotype&, double);
