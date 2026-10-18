import itertools,unittest
from .b_mask_bounds import transition_bounds,bounds_with_canvas


class BoundsTests(unittest.TestCase):
    def test_all_short_sequences_against_exhaustive_binary_completions(self):
        for n in range(1,7):
            for values in itertools.product((-1,0,1),repeat=n):
                positions=[i for i,v in enumerate(values)if v<0];counts=[]
                for replacements in itertools.product((0,1),repeat=len(positions)):
                    x=list(values)
                    for i,v in zip(positions,replacements):x[i]=v
                    counts.append(sum(a!=b for a,b in zip(x,x[1:])))
                result=transition_bounds(values)
                self.assertEqual((result['minimum'],result['maximum']),(min(counts),max(counts)))

    def test_outside_is_unobserved_and_never_bridged_in_canvas(self):
        r=bounds_with_canvas([0,0,-2,-2,1,1])
        self.assertEqual((r['whole_line']['minimum'],r['whole_line']['maximum']),(1,3))
        self.assertEqual((r['in_canvas_transition_minimum_sum'],r['in_canvas_transition_maximum_sum']),(0,0))
        self.assertEqual([(s['start_index'],s['end_index_inclusive'])for s in r['in_canvas_segments']],[(0,1),(4,5)])

    def test_ambiguous_samples_can_certify_lower_bound_without_assignment(self):
        r=transition_bounds([0,-1,1,-1,0])
        self.assertEqual((r['minimum'],r['maximum']),(2,2));self.assertEqual(r['classification'],'CONDITIONALLY_MULTIPLE')

    def test_no_sample_and_all_outside_are_explicit(self):
        self.assertIsNone(transition_bounds([])['minimum'])
        r=bounds_with_canvas([-2]*4);self.assertEqual(r['in_canvas_segments'],[])
        self.assertIsNone(r['in_canvas_transition_minimum_sum']);self.assertEqual(r['whole_line']['maximum'],3)


if __name__=='__main__':unittest.main()
