# PhysiCell custom scenarios

These directories contain only Leca-VC custom C++, headers, XML configuration,
and portable Makefiles. They require a separate PhysiCell 1.14.2 checkout:

```bash
make -C physicell/gse2565 PHYSICELL_ROOT=/path/to/PhysiCell
```

Generated binaries, object files, and the upstream PhysiCell tree are not
tracked. Runtime-specific `control.h` files may be generated from locked
configuration by the experiment orchestrator before compilation.

