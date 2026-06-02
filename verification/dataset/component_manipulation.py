from utils import process_label, create_out_string
import random
import pandas as pd

class ComponentManipulation():

    # Label scheme:
    #   0  allow_deny  – flip decision
    #   1  csub        – change subject
    #   2  cact        – change action
    #   3  cres        – change resource
    #   4  msub        – mask subject  → 'none'
    #   5  mres        – mask resource → 'none'
    #   6  mrules      – drop a rule from a multi-rule policy
    #   7  correct     – unmodified (positive class)

    def __init__(self, df) -> None:

        self.augs    = {'allow_deny': 0, 'csub': 0, 'cact': 0, 'cres': 0,
                        'msub': 0, 'mres': 0}
        self.augs2id = {'allow_deny': 0, 'csub': 1, 'cact': 2, 'cres': 3,
                        'msub': 4, 'mres': 5}
        self.id2augs = {0: 'allow_deny', 1: 'csub', 2: 'cact', 3: 'cres',
                        4: 'msub',       5: 'mres'}

        self.correct_acps = df[df['labels'] == 1]
        self.subs, self.acts, self.ress = self.collect_components()

    def collect_components(self):
        subs, acts, ress = [], [], []
        for policy in self.correct_acps['outputs'].to_list():
            for rule in process_label([policy]):
                s, a, r = rule['subject'], rule['action'], rule['resource']
                if s != 'none' and s not in subs:
                    subs.append(s)
                if a != 'none' and a not in acts:
                    acts.append(a)
                if r != 'none' and r not in ress:
                    ress.append(r)
        return subs, acts, ress

    # ------------------------------------------------------------------

    @staticmethod
    def _pick_different(pool: list, current: str) -> str | None:
        """Return a random pool element that differs from *current*, or None."""
        candidates = [v for v in pool if v != current]
        if not candidates:
            return None
        return candidates[random.randrange(len(candidates))]

    def change_rule(self, rule: dict, aug: str,
                    _tried: frozenset = frozenset()) -> tuple:
        """Apply *aug* to *rule* and return (rule, aug_applied).
        Falls back to an untried aug when the requested one cannot be applied.
        """
        if aug == 'allow_deny':
            rule['decision'] = 'deny' if 'allow' in rule['decision'] else 'allow'

        elif aug == 'csub':
            val = self._pick_different(self.subs, rule['subject'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['subject'] = val

        elif aug == 'cact':
            val = self._pick_different(self.acts, rule['action'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['action'] = val

        elif aug == 'cres':
            val = self._pick_different(self.ress, rule['resource'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['resource'] = val

        elif aug == 'msub' and rule['subject'] != 'none':
            rule['subject'] = 'none'
        elif aug == 'mres' and rule['resource'] != 'none':
            rule['resource'] = 'none'

        else:
            return self._fallback(rule, aug, _tried)

        return rule, aug

    def _fallback(self, rule: dict, failed_aug: str,
                  _tried: frozenset) -> tuple:
        _tried = _tried | {failed_aug}
        remaining = [a for a in self.augs if a not in _tried]
        if not remaining:
            raise RuntimeError(
                f"change_rule: no applicable augmentation for rule {rule}. "
                f"All tried: {_tried}"
            )
        return self.change_rule(rule, remaining[random.randrange(len(remaining))],
                                _tried)

    # ------------------------------------------------------------------

    def get_summary(self, df: pd.DataFrame):
        return df.groupby('labels')['labels'].count()

    def augment(self, num_times=5, missing_rule_count=4, print_summary=True):

        sents, npols, labels = [], [], []

        # ── Diagnostic ────────────────────────────────────────────────────
        parse_failures = 0
        for i, p in enumerate(self.correct_acps["outputs"].head(3)):
            parsed = process_label([p])
            print(f"  [diag] sample {i}: output={str(p)[:100]!r}")
            print(f"          -> parsed={parsed}")
        total = len(self.correct_acps)
        for p in self.correct_acps["outputs"]:
            if not process_label([p]):
                parse_failures += 1
        print(f"  [diag] parse failures: {parse_failures}/{total} correct ACP rows\n")

        # ── Field-level augmentation ──────────────────────────────────────
        aug_keys = list(self.augs.keys())   # fixed order for randrange indexing
        for _ in range(num_times):
            for s, p in zip(self.correct_acps['inputs'],
                            self.correct_acps['outputs']):
                pols = process_label([p])
                if not pols:
                    continue

                randpol = random.randrange(len(pols))
                modified = []
                naug = None
                for i, rule in enumerate(pols):
                    if i == randpol:
                        aug = aug_keys[random.randrange(len(aug_keys))]
                        mod_rule, naug = self.change_rule(rule, aug)
                        modified.append(mod_rule)
                        self.augs[naug] += 1
                    else:
                        modified.append(rule)

                sents.append(s)
                npols.append(create_out_string(modified))
                labels.append(self.augs2id[naug])

        # ── Missing-rule augmentation (label 6) ───────────────────────────
        self.augs['mrules'] = 0
        self.augs2id['mrules'] = 6
        self.id2augs[6] = 'mrules'

        for _ in range(missing_rule_count):
            for s, p in zip(self.correct_acps['inputs'],
                            self.correct_acps['outputs']):
                pols = process_label([p])
                if len(pols) > 1:
                    pols.pop(random.randrange(len(pols)))
                    sents.append(s)
                    npols.append(create_out_string(pols))
                    labels.append(6)
                    self.augs['mrules'] += 1

        # ── Correct / positive class (label 7) ───────────────────────────
        csents = self.correct_acps['inputs'].to_list()
        cpols  = self.correct_acps['outputs'].to_list()
        sents.extend(csents)
        npols.extend(cpols)
        labels.extend([7] * len(csents))

        df = pd.DataFrame({'inputs': sents, 'outputs': npols, 'labels': labels})

        if print_summary:
            print(self.get_summary(df))

        return df