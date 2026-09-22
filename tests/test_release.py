from __future__ import annotations
import ast, json, os, unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from lecavc.llm_client import MissingAPIKeyError, require_api_key
from lecavc.prompt_firewall import PromptFirewallError, enforce_messages
from lecavc.schemas import NumericField, ProgramSchema, SchemaValidationError, validate_program
from workflows.run_benchmark import preflight

ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_manifest_has_all_figures(self):
        figures=json.loads((ROOT/"manifests/paper_figures.yaml").read_text())["figures"]
        self.assertEqual(set(figures),{f"fig{i}" for i in range(1,7)}|{f"figS{i}" for i in range(1,9)})

    def test_firewall_blocks_identifiers(self):
        with self.assertRaises(PromptFirewallError): enforce_messages([{"role":"user","content":"Predict GSE120575 Post outcome"}])
        enforce_messages([{"role":"user","content":"Use normalized step 0 and the local numeric state."}])

    def test_schema_type_and_range(self):
        schema=ProgramSchema({"communication.signal":NumericField(0,0.5)})
        self.assertEqual(validate_program({"dominant_program":"repair","communication":{"signal":.2}},schema)["dominant_program"],"repair")
        with self.assertRaises(SchemaValidationError): validate_program({"dominant_program":"repair","communication":{"signal":.8}},schema)

    def test_missing_key_never_falls_back(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(MissingAPIKeyError): require_api_key()

    def test_probe_frozen_result(self):
        calls=pd.read_csv(ROOT/"source_data/figS7/deidentified_exact_runtime_call_results.csv")
        self.assertEqual((len(calls),int(calls.heldout_answer_correct.sum()),int(calls.abstained_unknown.sum())),(30,1,23))

    def test_no_large_or_compiled_files(self):
        bad=[]
        for path in ROOT.rglob("*"):
            if path.is_symlink() or not path.is_file() or "build" in path.parts or "__pycache__" in path.parts: continue
            if path.stat().st_size>50*1024*1024 or path.suffix in {".pyc",".o",".so",".exe"} or path.name.endswith("Zone.Identifier"): bad.append(str(path.relative_to(ROOT)))
        self.assertEqual(bad,[])

    def test_fibrosis_application_remains_fail_closed(self):
        report=preflight("fibrosis_application",require_online=False)
        self.assertEqual(report["status"],"BLOCKED")
        self.assertTrue(any("paused" in item for item in report["failures"]))

    def test_fibrosis_delivery_provenance(self):
        audit=json.loads((ROOT/"experiments/fibrosis_application/provenance/delivery_audit.json").read_text())
        self.assertEqual(audit["source_archive"]["sha256"],"c57ae54b1615158357bac856d0f3de3281ee88f6eb4bc08be0d59e3488ce7abc")
        self.assertFalse(audit["integration"]["experiment_executed_during_integration"])

    def test_fig6_frozen_reference_integrity(self):
        import hashlib
        from PIL import Image
        path=ROOT/"source_data/fig6/figure6_frozen_reference.png"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),"a1d7e849d42c56eb7010f033e5c5103b97c963c4fef2d3f36443ccbf62dcc3cf")
        with Image.open(path) as image:
            self.assertEqual(image.size,(7200,3600))
        provenance=json.loads((ROOT/"source_data/fig6/provenance.json").read_text())
        self.assertFalse(provenance["scientific_evidence_review"]["full_run_success_accepted"])

    def test_supported_code_dependency_contract(self):
        manifest=json.loads((ROOT/"manifests/code_entrypoints.json").read_text())
        missing=[]
        for workflow,record in manifest["workflows"].items():
            for relative in [record["entrypoint"],*record.get("required_paths",[])]:
                if not (ROOT/relative).exists(): missing.append(f"{workflow}: {relative}")
        self.assertEqual(missing,[])

    def test_python_sources_parse_without_bytecode(self):
        failures=[]
        for path in ROOT.rglob("*.py"):
            if any(part in {"build","__pycache__"} for part in path.parts): continue
            try: ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
            except (SyntaxError,UnicodeDecodeError) as error: failures.append(f"{path.relative_to(ROOT)}: {error}")
        self.assertEqual(failures,[])

    def test_no_legacy_local_physicell_checkout_name(self):
        offenders=[]
        for base in (ROOT/"experiments",ROOT/"scripts",ROOT/"workflows"):
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix not in {".py",".sh"}: continue
                if "PhysiCell-master" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders,[])

if __name__=="__main__": unittest.main()
