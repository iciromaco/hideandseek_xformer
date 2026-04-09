import json
import numpy as np

JPATH = 'experiments/signed_forward.json'

def main():
    j = json.load(open(JPATH))
    recs = j.get('records_sample', [])
    if not recs:
        print('No sample records found in', JPATH)
        return
    a = np.array([r['action'][0] for r in recs], dtype=np.float32)
    s = np.array([r['signed_forward'] for r in recs], dtype=np.float32)
    ap = np.array([r.get('applied_forward', 0.0) for r in recs], dtype=np.float32)
    corr_a_s = float(np.corrcoef(a, s)[0, 1])
    corr_ap_s = float(np.corrcoef(ap, s)[0, 1])
    print('corr a0 vs signed_forward:', corr_a_s)
    print('corr applied_forward vs signed_forward:', corr_ap_s)
    print('mean a0', float(a.mean()), 'mean applied', float(ap.mean()), 'mean signed_forward', float(s.mean()))

if __name__ == '__main__':
    main()
