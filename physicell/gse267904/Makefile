PHYSICELL_ROOT ?= ../../vendor/PhysiCell
ROOT := $(PHYSICELL_ROOT)
CXX:=g++
CXXFLAGS:=-O3 -fopenmp -m64 -std=c++11 -I$(ROOT)
CORE:=$(ROOT)/BioFVM_vector.o $(ROOT)/BioFVM_mesh.o $(ROOT)/BioFVM_microenvironment.o $(ROOT)/BioFVM_solvers.o $(ROOT)/BioFVM_matlab.o $(ROOT)/BioFVM_utilities.o $(ROOT)/BioFVM_basic_agent.o $(ROOT)/BioFVM_MultiCellDS.o $(ROOT)/BioFVM_agent_container.o $(ROOT)/pugixml.o $(ROOT)/PhysiCell_phenotype.o $(ROOT)/PhysiCell_cell_container.o $(ROOT)/PhysiCell_standard_models.o $(ROOT)/PhysiCell_cell.o $(ROOT)/PhysiCell_custom.o $(ROOT)/PhysiCell_utilities.o $(ROOT)/PhysiCell_constants.o $(ROOT)/PhysiCell_basic_signaling.o $(ROOT)/PhysiCell_signal_behavior.o $(ROOT)/PhysiCell_rules.o $(ROOT)/PhysiCell_SVG.o $(ROOT)/PhysiCell_pathology.o $(ROOT)/PhysiCell_MultiCellDS.o $(ROOT)/PhysiCell_various_outputs.o $(ROOT)/PhysiCell_pugixml.o $(ROOT)/PhysiCell_settings.o $(ROOT)/PhysiCell_geometry.o
gse267904_spatial_commot_agent: custom.o $(CORE)
	$(CXX) $(CXXFLAGS) -o $@ $(CORE) custom.o $(ROOT)/main.cpp
custom.o: custom.cpp custom.h control.h
	$(CXX) $(CXXFLAGS) -c custom.cpp
