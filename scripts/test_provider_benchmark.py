"""CPU-only checks for payload isolation and response validation."""
import json
from types import SimpleNamespace
import unittest
from benchmark_emails import DEFAULT_DATASET, load_dataset
from benchmark_providers import native_request, validate_choice, trial_summary

class ProviderBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.categories, cls.guide, _ = load_dataset(DEFAULT_DATASET)

    def test_reference_labels_never_sent(self):
        row = dict(self.rows[0], reference_category='SECRET_REFERENCE_LABEL', reference_reason='SECRET_REASON')
        for body in [native_request(row,'jev','jev-1.13.0',self.categories,self.guide,True),
                     native_request(row,'clef','clef',self.categories,self.guide,True)]:
            encoded=json.dumps(body)
            self.assertNotIn('SECRET_REFERENCE_LABEL',encoded)
            self.assertNotIn('SECRET_REASON',encoded)
        jev=native_request(row,'jev','jev-1.13.0',self.categories,self.guide,True)
        self.assertEqual(set(jev),{'model','state','questions'})
        self.assertEqual(jev['state']['classification_guide'],self.guide)
        generic=native_request(row,'systemone','laya',self.categories,self.guide,True)
        self.assertEqual(set(generic),{'model','state','questions'})
        self.assertEqual(generic['state'],jev['state'])

    def test_invalid_probabilities_rejected(self):
        p={k:0.0 for k in self.categories};p['orders']=1.0
        answer={'answers':{'category':{'choice':'orders','probabilities':p}}}
        self.assertEqual(validate_choice(answer,self.categories)[0],'orders')
        p['orders']=float('nan')
        with self.assertRaises(ValueError):validate_choice(answer,self.categories)
        p['orders']=1.0;answer['answers']['category']['choice']='marketing'
        with self.assertRaises(ValueError):validate_choice(answer,self.categories)

    def test_vendor_two_decimal_rounding_is_preserved(self):
        p={k:0.0 for k in self.categories};p['personal']=.94;p['other']=.05
        answer={'answers':{'category':{'choice':'personal','probabilities':p}}}
        with self.assertRaises(ValueError):validate_choice(answer,self.categories)
        choice, returned=validate_choice(answer,self.categories,rounded_probabilities=True)
        self.assertEqual(choice,'personal')
        self.assertIs(returned,p)
        p['personal']=.4
        with self.assertRaises(ValueError):validate_choice(answer,self.categories,rounded_probabilities=True)

    def test_errors_excluded_from_latency_but_counted(self):
        good={'id':'sample-001','reference_category':'orders','choice':'orders','client_wall_ms':10,
              'usage':{},'provider_usage':{'input_tokens':20}}
        bad={'id':'sample-002','reference_category':'orders','error':'HTTPStatusError','client_wall_ms':200}
        summary=trial_summary([good,bad],.21,self.categories)
        self.assertEqual(summary['errors'],1)
        self.assertEqual(summary['reference_matches'],1)
        self.assertEqual(summary['client_wall_ms']['median'],10)
        self.assertEqual(summary['total_input_tokens'],20)

if __name__=='__main__':unittest.main()
