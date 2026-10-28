"""Registered input checks plus trusted C150 tensor/gate reader regression tests."""
from core152 import *
import tempfile,unittest
from inputs152 import load
from fixture152 import dummy_inputs
# These inherited tests exercise origin rows, route caches, final step identity and native gates.
from test_bindings150 import BindingChecks as PriorOriginBindingChecks

class CurrentInputChecks(unittest.TestCase):
    def test_wrong_current_production_pin_blocks_before_archive_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);paths,_=dummy_inputs(p/'inputs')
            with self.assertRaises(ValueError):load(paths,p/'out')
    def test_seven_originals_and_no_C151_result_ZIP_required(self):
        self.assertEqual(set(INPUT_KEYS),{'C150','C149','C148','C147','C146','C145','C143'});self.assertFalse(design()['C151_original_result_zip_required'])
if __name__=='__main__':unittest.main(verbosity=2)
