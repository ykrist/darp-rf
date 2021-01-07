from darp import *
from oru.grb import BinVarDict, IntVar, CtsVar
from oru import slurm
from gurobi import *
from typing import Tuple
from utils.data import modify, DARP_Data
from collections import OrderedDict

F_DBG = ResFrag(start=LNode(loc=5, load=frozenset()), end=LNode(loc=28, load=frozenset()), path=(5, 12, 4, 29, 36, 28))

HEURISTIC_MODEL_PARAMETERS = OrderedDict({
    'OutputFlag': 0,
    'TimeLimit': 5,
    'LazyConstraints': 1,
    'GURO_PAR_MINBPFORBID' : 1,
})

MASTER_MODEL_PARAMETERS = OrderedDict({
    'LazyConstraints': 1,
    'PreCrush': 1,
    'GURO_PAR_MINBPFORBID' : 1,
    'VarBranch' : 1,
    'MIRCuts' : 2
})

Chain = Tuple[ResFrag]

@dataclasses.dataclass(frozen=True)
class DominationInfo:
    total_travel_time: float
    total_travel_cost: float
    earliest_schedule: Tuple[float, ...]
    latest_schedule: Tuple[float, ...]
    earliest_late_schedule: Tuple[float, ...]
    latest_early_schedule: Tuple[float, ...]
    # these should be sorted by job no. they correspond to, in order to be consistent across different fragments
    partial_job_idxs: Tuple[int, ...]


class Dominator:
    def __init__(self, fragments, data: DARP_Data):
        self.data = data
        self.fragments = fragments
        self.dominfo: Dict[ResFrag, DominationInfo] = {}

    def get_undominated_fragments(self):
        domdict = defaultdict(set)

        for f in self.fragments:
            if f.end.load >= f.start.load:
                partial_job_idxs = tuple(f.path.index(i) for i in sorted(f.end.load - f.start.load))
                dom_crit = self.domination_criterion_no_unmatched_deliveries
            elif f.end.load < f.start.load:
                partial_job_idxs = tuple(f.path.index(i + self.data.n) for i in sorted(f.start.load - f.end.load))
                dom_crit = self.domination_criterion_no_unmatched_pickups
            else:
                domdict[None].add(f)
                continue

            earliest_schedule = get_early_schedule(f.path, data)
            latest_schedule = get_late_schedule(f.path, data)
            assert latest_schedule is not None
            earliest_late_schedule = get_late_schedule(f.path, data, end_time=earliest_schedule[-1])
            latest_early_schedule = get_early_schedule(f.path, data, start_time=latest_schedule[0])
            assert earliest_late_schedule is not None
            assert latest_early_schedule is not None

            domkey = (f.start, f.end, frozenset(f.path))
            # noinspection PyArgumentList
            dominfo_f = DominationInfo(
                total_travel_time=sum(self.data.travel_time[i, j] for i, j in zip(f.path, f.path[1:])),
                total_travel_cost=sum(self.data.travel_cost[i, j] for i, j in zip(f.path, f.path[1:])),
                earliest_schedule=earliest_schedule,
                latest_schedule=latest_schedule,
                earliest_late_schedule=earliest_late_schedule,
                latest_early_schedule=latest_early_schedule,
                partial_job_idxs=partial_job_idxs,
            )
            self.dominfo[f] = dominfo_f

            dominated = set()
            for g in domdict[domkey]:
                if dom_crit(g, f):
                    assert len(dominated) == 0
                    break
                elif dom_crit(f, g):
                    dominated.add(g)
            else:
                # f is undominated
                domdict[domkey] -= dominated
                domdict[domkey].add(f)

        return set(itertools.chain(*domdict.values()))

    def domination_criterion_no_unmatched_deliveries(self, f: ResFrag, g: ResFrag):
        """Returns true if f dominates g, and false otherwise."""
        F: DominationInfo = self.dominfo[f]
        G: DominationInfo = self.dominfo[g]

        if F.total_travel_time > G.total_travel_time + EPS:
            return False

        if F.total_travel_cost > G.total_travel_cost + EPS:
            return False

        # check Ef <= Eg
        if F.earliest_schedule[-1] > G.earliest_schedule[-1] + EPS:
            return False

        # check Lf >= Lg
        if F.latest_schedule[0] + EPS < G.latest_schedule[0]:
            return False

        for pf, pg in zip(F.partial_job_idxs, G.partial_job_idxs):
            if F.earliest_late_schedule[pf] + EPS < G.earliest_late_schedule[pg]:
                return False

            if F.latest_schedule[pf] + EPS < G.latest_schedule[pg]:
                return False

        return True

    def domination_criterion_no_unmatched_pickups(self, f: ResFrag, g: ResFrag):
        """Returns true if f dominates g, and false otherwise."""
        F: DominationInfo = self.dominfo[f]
        G: DominationInfo = self.dominfo[g]

        if F.total_travel_time > G.total_travel_time + EPS:
            return False

        if F.total_travel_cost > G.total_travel_cost + EPS:
            return False

        # check Ef <= Eg
        if F.earliest_schedule[-1] > G.earliest_schedule[-1] + EPS:
            return False

        # check Lf >= Lg
        if F.latest_schedule[0] + EPS < G.latest_schedule[0]:
            return False

        for pf, pg in zip(F.partial_job_idxs, G.partial_job_idxs):
            if F.earliest_schedule[pf] > EPS + G.earliest_schedule[pg]:
                return False

            if F.latest_early_schedule[pf] > EPS + G.latest_early_schedule[pg]:
                return False

        return True


def calc_path_cost(path, data: DARP_Data):
    return sum(data.travel_cost[i, j] for i, j in zip(path, path[1:]))


class DebugVizNetwork:
    def __init__(self, data):
        self.nodes = list()
        self._nodes_ids = set()
        self.edges = list()
        self.data = data

    def _convert_node(self, n: LNode):
        return (n.loc, tuple(sorted(n.load)))

    def add_node(self, n: LNode):
        nd = self._convert_node(n)
        if nd not in self._nodes_ids:
            self.nodes[nd] = {
                'id': nd,
                'loc': nd[0],
                'load': nd[1],
                'x': self.data.loc[n.loc][0],
                'y': self.data.loc[n.loc][1],
                'tw_start': self.data.tw_start[n.loc],
                'tw_end': self.data.tw_start[n.loc]
            }
        return nd

    def add_arc(self, arc: LArc, **attrs):
        s = self.add_node(arc.start)
        e = self.add_node(arc.end)
        self.edges.append({
            'start': s,
            'end': e,
            'type': 'a',
            'time': self.data.travel_time[arc.start.loc, arc.end.loc],
            **attrs
        })

    def add_fragment(self, f: ResFrag, **attrs):
        s = self.add_node(f.start)
        e = self.add_node(f.end)
        self.edges.append({
            'start': s,
            'end': e,
            'type': 'f',
            'time': sum(self.data.travel_time[i, j] for i, j in zip(f.path, f.path[1:])),
            'path': f.path,
            **attrs
        })

    def dump(self, filename):
        with open(filename, 'w') as fp:
            json.dump({
                'nodes': self.nodes,
                'edges': self.edges
            }, fp, indent='\t')



CUT_FUNC_LOOKUP = dict()
VALID_INEQUALITIES = set()
LAZY_CUTS = set()

def cut(name: str, valid_inequality : bool):
    def decorator(func):
        global CUT_FUNC_LOOKUP, VALID_INEQUALITIES, LAZY_CUTS
        assert name not in CUT_FUNC_LOOKUP, f"{name} is already defined."
        CUT_FUNC_LOOKUP[name] = func
        if valid_inequality:
            VALID_INEQUALITIES.add(name)
        else:
            LAZY_CUTS.add(name)
        return func
    return decorator


class RestrictedFragmentGenerator:
    def __init__(self, data: DARP_Data):
        self.loc_neighbours = defaultdict(set)
        for i, j in data.travel_cost:
            self.loc_neighbours[i].add(j)

        self.loc_neighbours.default_factory = None
        self.data = data
        self.o_depot_loc = 0
        self.d_depot_loc = 2*data.n + 1
        self.parent_paths = 0
        self.fragments = set()

        for i in self.data.P:
            t = max(data.tw_start[i], data.travel_time[self.o_depot_loc, i])
            self._extend(
                (i,),
                t,
                self.data.demand[i],
                {i + data.n: 0},
                data.travel_time[self.o_depot_loc, i]
            )

    def get_restricted_fragments(self, path):
        rf = set()
        for i in range(len(path) // 2):
            start_load = frozenset(path[:i])
            for j in range(len(path) // 2 + 1, len(path) + 1):
                end_load = frozenset(i - self.data.n for i in path[j:])
                rf.add(ResFrag(LNode(path[i], start_load), LNode(path[j - 1], end_load), path[i:j]))
        return rf

    def _extend(self, path, time, load, pending_deliveries: dict, min_route_time):
        i = path[-1]
        if time > self.data.tw_end[i]:
            return False
        if min_route_time > self.data.max_ride_time[0] - self.data.travel_time.get((i, self.o_depot_loc),0) + EPS:
            return False
        if load > self.data.capacity:
            return False
        for j, min_onboard_time in pending_deliveries.items():
            if min_onboard_time > self.data.max_ride_time[j-self.data.n] + self.data.travel_time.get((i, j), 0)+ EPS:
                return False

        if len(pending_deliveries) == 0:
            if self.check_maximum_ride_time(path):
                self.parent_paths += 1
                for f in self.get_restricted_fragments(path):
                    self.fragments.add(f)
                return True
            else:
                return False

        can_finish = False
        for j in pending_deliveries:
            if j not in self.loc_neighbours[i]:
                continue

            new_pending_deliveries = {jd: t + self.data.travel_time[i, j]
                                      for jd, t in pending_deliveries.items() if jd != j}
            tt = self.data.travel_time[i, j]
            can_finish |= self._extend(
                path + (j,),
                max(time + tt, self.data.tw_start[j]),
                load + self.data.demand[j],
                new_pending_deliveries,
                min_route_time+tt
            )

        if can_finish and i not in self.data.D:
            for j in self.loc_neighbours[i]:
                if j in path or j not in self.data.P:
                    continue

                new_pending_deliveries = {jd: t + self.data.travel_time[i, j]
                                          for jd, t in pending_deliveries.items()}
                new_pending_deliveries[j + self.data.n] = 0
                tt = self.data.travel_time[i, j]
                self._extend(path + (j,),
                             max(time + tt, self.data.tw_start[j]),
                             load + self.data.demand[j],
                             new_pending_deliveries,
                             min_route_time+tt
                             )

        return can_finish

    def check_maximum_ride_time(self, path):
        s = get_early_schedule(path, self.data)
        return (s is not None)


def is_min_frag_to(f):
    return len(f.end.load) > 0 and len(f.end.load) + 2 == len(f.path)


def is_min_frag_from(f):
    return len(f.start.load) > 0 and len(f.start.load) + 2 == len(f.path)


class NetworkRestriction:
    def __init__(self, network, fragments):
        self.network = network
        assert network.restriction is None
        self.fragments = fragments

        self.p_nodes = set(map(lambda f: f.start, fragments))
        self.d_nodes = set(map(lambda f: f.end, fragments))

        arcs_set_a = self.network.end_arcs.copy()
        for p in self.p_nodes:
            arcs_set_a.update(self.network.arcs_by_end[p])  # this includes start arcs

        arcs_set_b = self.network.start_arcs.copy()
        for d in self.d_nodes:
            arcs_set_b.update(self.network.arcs_by_start[d])

        self.arcs = arcs_set_a & arcs_set_b

    def __enter__(self):
        self.network.restriction = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.network.restriction = None


class Network:
    def __init__(self, data: DARP_Data, fragment_domination=True, filter_fragments=True):
        self.data = data
        self.o_depot = LNode(0, frozenset())
        self.d_depot = LNode(data.n * 2 + 1, frozenset())
        self.min_fragments_from = defaultdict(set)
        self.min_fragments_to = defaultdict(set)
        self.fragments_by_end = defaultdict(set)
        self.fragments_by_start = defaultdict(set)
        self.fragments = set()
        self.fragments_by_loc = defaultdict(set)
        self.fragments_by_locset = defaultdict(set)
        self.filter_fragments = filter_fragments
        self.fragment_domination = fragment_domination
        self.path_time = dict()
        self.path_cost = dict()

        self.p_nodes = set()
        self.d_nodes = set()
        self.d_nodes_by_load = defaultdict(set)
        self.p_nodes_by_load = defaultdict(set)

        self.arcs = set()
        self.start_arcs = set()
        self.end_arcs = set()
        self.dp_arcs = set()
        self.arcs_by_end = defaultdict(set)
        self.arcs_by_start = defaultdict(set)

        self.arc_cost = dict()
        self.arc_time = dict()
        self.network_size_info = dict()

        self.max_fragment_length = None

        self._is_legal_cache = dict()
        self._is_legal_cache_info = {'total': 0, 'misses': 0}

        # Network restriction attributes.
        # r_fragments restricts everything about the model - lazy constraints, valid inequalities, and heuristics
        # will only use fragments in r_fragments.
        # r_allowed_fragments only restricts the generation of heuristic solutions.
        # The two may be used together - if f is in r_fragments but not in r_allowed_fragments, then any heuristic
        # solutions will not contain f, but f may be present in VIs or LCs.  r_allowed_fragments should always be a
        # subset of r_fragments - think of it like setting some fragments to zero but keeping them in the model.
        self.restriction = None

        self.legal_fragments_after_fragment_lookup = {}
        self.legal_fragments_before_fragment_lookup = {}


    def restrict(self, fragments):
        return NetworkRestriction(self, fragments)

    @memoise
    def path_loc_pairs(self, path):
        # non_depot_locs = [i for i in path if i != self.o_depot.loc and i != self.d_depot.loc]
        return list(map(frozenset, itertools.combinations(path, 2)))

    @memoise
    def path_loc_triples(self, path):
        # non_depot_locs = [i for i in path if i != self.o_depot.loc and i != self.d_depot.loc]
        return list(map(frozenset, itertools.combinations(path, 3)))

    @memoise
    def get_fragments_by_coverset(self, locset: frozenset):
        """ Returns all fragments whose locations are entirely contained within `locset`. """

        n = len(locset)
        try:
            frags = self.fragments_by_locset[locset]
        except KeyError:
            frags = set()

        if n > 2:
            subset_size = min(n - 1, self.max_fragment_length)  # overwise for large locsets, the exp. runtime bites.
            for s in itertools.combinations(locset, subset_size):
                frags |= self.get_fragments_by_coverset(frozenset(s))

        return frags

    def get_triad(self, triad: FrozenSet[int]):
        i1, i2, i3 = triad
        F1 = self.fragments_by_loc[i1]
        F2 = self.fragments_by_loc[i2]
        F3 = self.fragments_by_loc[i3]
        return (F1 & F2) | (F1 & F3) | (F2 & F3)

    def build(self):
        print('Generating fragments...')
        frag_generator = RestrictedFragmentGenerator(self.data)
        self.fragments = frag_generator.fragments
        n_frags = len(self.fragments)
        self.network_size_info['generation_fragments'] = n_frags
        self.network_size_info['generation_parent_paths'] = frag_generator.parent_paths
        print(f'{n_frags:,d} fragments generated from {frag_generator.parent_paths:,d} parent paths.')

        if self.filter_fragments:
            print('Filtering fragments with partial loads...')
            min_fragments_by_start_load = defaultdict(set)
            min_fragments_by_end_load = defaultdict(set)
            fragments_by_loc = defaultdict(set)
            min_fragments = set()
            for f in self.fragments:
                if len(f.end.load) == 0 and len(f.path) == len(f.start.load) + 2:
                    min_fragments.add(f)
                    min_fragments_by_start_load[f.start.load].add(f)
                if len(f.start.load) == 0 and len(f.path) == len(f.end.load) + 2:
                    min_fragments_by_end_load[f.end.load].add(f)
                    min_fragments.add(f)
                for i in f.path:
                    fragments_by_loc[i].add(f)

            illegal_fragments = set()
            skip_check_fragments = set()
            cnt_total = [0, 0, 0]
            cnt_illegal = [0, 0, 0]
            num_illegal_minimal_fragments = 0
            print(f'{len(min_fragments):,d}/{len(self.fragments):,d} are minimal fragments.')
            for is_min_frag, f in itertools.chain(zip(itertools.repeat(True),min_fragments),
                                     zip(itertools.repeat(False), self.fragments - min_fragments)):
                empty_start_load = len(f.start.load) == 0
                empty_end_load = len(f.end.load) == 0

                fset = set(f.path)
                if not empty_start_load and not empty_end_load:
                    cnt_total[0] += 1
                    # f can't be skip_check_fragments because f is not minimal.
                    for g1 in min_fragments_by_end_load[f.start.load]:
                        if (
                                g1.end.loc not in fset and
                                (g1.path[-1], f.path[0]) in self.data.travel_time and
                                self.is_legal((g1, f), complete=False, check_cover=False)
                        ):
                            for g2 in min_fragments_by_start_load[f.end.load]:
                                if (
                                        g2.start.loc != g1.end.loc - self.data.n and
                                        g2.start.loc not in fset and
                                        (f.path[-1], g2.path[0]) in self.data.travel_time and
                                        self.is_legal((g1, f, g2), complete=True, check_cover=False)
                                ):
                                    self.dp_arcs.add(LArc(g1.end, f.start))
                                    self.dp_arcs.add(LArc(f.end, g2.start))
                                    skip_check_fragments.add(g1)
                                    skip_check_fragments.add(g2)
                                    break
                            else:
                                continue
                            break
                    else:
                        assert not is_min_frag
                        illegal_fragments.add(f)
                        cnt_illegal[0] += 1

                elif not empty_start_load and empty_end_load:
                    cnt_total[1] += 1

                    if f not in skip_check_fragments:
                        for g in min_fragments_by_end_load[f.start.load]:
                            if (
                                    g.end.loc not in fset and
                                    (g.path[-1], f.path[0]) in self.data.travel_time and
                                    self.is_legal((g, f), complete=True, check_cover=False)
                            ):
                                skip_check_fragments.add(g)
                                self.dp_arcs.add(LArc(g.end, f.start))
                                break
                        else:
                            illegal_fragments.add(f)
                            if is_min_frag:
                                min_fragments_by_start_load[f.start.load].remove(f)
                                num_illegal_minimal_fragments += 1
                            cnt_illegal[1] += 1

                elif empty_start_load and not empty_end_load:
                    cnt_total[2] += 1

                    if f not in skip_check_fragments:
                        for g in min_fragments_by_start_load[f.end.load]:
                            if (
                                    g.start.loc not in fset and
                                    (f.path[-1], g.path[0]) in self.data.travel_time and
                                    self.is_legal((f, g), complete=True, check_cover=False)
                            ):
                                skip_check_fragments.add(g)
                                self.dp_arcs.add(LArc(f.end, g.start))
                                break
                        else:
                            illegal_fragments.add(f)
                            if is_min_frag:
                                num_illegal_minimal_fragments += 1
                                min_fragments_by_end_load[f.end.load].remove(f)
                            cnt_illegal[2] += 1

            print(f'{num_illegal_minimal_fragments:,d}/{len(min_fragments):,d} of minimal fragments were removed.')

            output = TablePrinter(["Has start load", "Has end load", "Inital", "Removed", "Remaining", "Removed %"],
                                  min_col_width=15)
            output.print_line("o", "o", cnt_total[0], cnt_illegal[0], cnt_total[0] - cnt_illegal[0],
                              "{:.1f}%".format(100 * cnt_illegal[0] / (EPS + cnt_total[0])))
            output.print_line("o", "", cnt_total[1], cnt_illegal[1], cnt_total[1] - cnt_illegal[1],
                              "{:.1f}%".format(100 * cnt_illegal[1] / (EPS + cnt_total[1])))
            output.print_line("", "o", cnt_total[2], cnt_illegal[2], cnt_total[2] - cnt_illegal[2],
                              "{:.1f}%".format(100 * cnt_illegal[2] / (EPS+cnt_total[2])))
            n_complete_frags = len(self.fragments) - sum(cnt_total)
            output.print_line("", "", n_complete_frags, "-", "-", "-")
            cnt_total = sum(cnt_total) + n_complete_frags
            cnt_illegal = sum(cnt_illegal)
            output.print_line("total:", "", cnt_total, cnt_illegal, cnt_total - cnt_illegal,
                              "{:.1f}%".format(100 * cnt_illegal / (EPS+cnt_total)))

            self.network_size_info['filtering_fragments_removed'] = cnt_illegal
            self.fragments -= illegal_fragments


        n_frags = len(self.fragments)

        if self.fragment_domination:
            print('Dominating fragments...')
            dominator = Dominator(self.fragments, self.data)
            undominated_fragments = dominator.get_undominated_fragments()
            n_undom_frags = len(undominated_fragments)
            n_dom_frags = n_frags - n_undom_frags
            print(f"{n_dom_frags:,d} ({(100 * n_dom_frags / n_frags):.1f}%) fragments removed.")
            print(f"{n_undom_frags:,d} fragments remain.")
            self.network_size_info['domination_fragments_removed'] = n_dom_frags
            self.fragments = undominated_fragments

        else:
            self.network_size_info['dominated_fragments'] = 0

        for f in self.fragments:
            if f.path not in self.path_cost:
                self.path_cost[f.path] = calc_path_cost(f.path, data)

        if self.filter_fragments:
            self.add_arcs_and_nodes()
        else:
            self.add_arcs_and_nodes_part1()
            self.network_reduce()
            self.add_arcs_and_nodes_part2()

        self.finalise_lookups()
        self.freeze_dicts()

    def add_arcs_and_nodes(self):
        # Build set of nodes
        for f in self.fragments:
            if len(f.end.load) == 0 and len(f.path) == len(f.start.load) + 2:
                self.min_fragments_from[f.start].add(f)
            if len(f.start.load) == 0 and len(f.path) == len(f.end.load) + 2:
                self.min_fragments_to[f.end].add(f)
            self.p_nodes.add(f.start)
            self.d_nodes.add(f.end)

        self.min_fragments_from.default_factory = None
        self.min_fragments_to.default_factory = None

        for f in self.fragments:
            self.fragments_by_start[f.start].add(f)
            self.fragments_by_end[f.end].add(f)

        for d in self.d_nodes:
            for p in self.p_nodes:
                if d.load == p.load and (d.loc, p.loc) in self.data.travel_time:
                    a = LArc(d, p)
                    if a in self.dp_arcs:  # some arcs have been previously "discovered" during fragment filtering.
                        continue

                    candidate_pairs = ((f, g) for f in self.min_fragments_to[d] for g in self.min_fragments_from[p])
                    for f, g in candidate_pairs:
                        if f.end.loc != g.start.loc + self.data.n and \
                                self.is_legal((f, g), complete=True, check_cover=False):
                            self.dp_arcs.add(a)
                            break

        for a in self.dp_arcs:
            self.arcs_by_end[a.end].add(a)
            self.arcs_by_start[a.start].add(a)

        self.start_arcs = {LArc(self.o_depot, p) for p in self.p_nodes if len(p.load) == 0}

        for a in self.start_arcs:
            self.arcs_by_end[a.end].add(a)
        self.arcs_by_start[self.o_depot] = self.start_arcs

        self.end_arcs = {LArc(d, self.d_depot) for d in self.d_nodes if len(d.load) == 0}
        for a in self.end_arcs:
            self.arcs_by_start[a.start].add(a)
        self.arcs_by_end[self.d_depot] = self.end_arcs

        self.arcs = self.start_arcs | self.dp_arcs | self.end_arcs

        table = [
            ('Start Arcs', len(self.start_arcs)),
            ('End Arcs', len(self.end_arcs)),
            ('DP Arcs', len(self.dp_arcs)),
            ('Arcs', len(self.arcs)),
            ('Fragments', len(self.fragments)),
            ('D Nodes', len(self.d_nodes)),
            ('P Nodes', len(self.p_nodes)),
            ('Nodes', len(self.d_nodes) + len(self.p_nodes) + 2),
        ]
        output = TablePrinter(['Final network size:', ''], min_col_width=8)
        for row in table:
            output.print_line(*row)
        for k, v in table:
            k = 'final_size_' + k.lower().replace(' ', '_')
            self.network_size_info[k] = v

    def add_arcs_and_nodes_part1(self):
        # Build set of nodes
        for f in self.fragments:
            if len(f.end.load) == 0 and len(f.path) == len(f.start.load) + 2:
                self.min_fragments_from[f.start].add(f)
            if len(f.start.load) == 0 and len(f.path) == len(f.end.load) + 2:
                self.min_fragments_to[f.end].add(f)
            self.p_nodes.add(f.start)
            self.d_nodes.add(f.end)

        self.min_fragments_from.default_factory = None
        self.min_fragments_to.default_factory = None

        for f in self.fragments:
            self.fragments_by_start[f.start].add(f)
            self.fragments_by_end[f.end].add(f)

        for d in self.d_nodes:
            for p in self.p_nodes:
                if d.load == p.load and (d.loc, p.loc) in self.data.travel_time:
                    a = LArc(d, p)
                    if a in self.dp_arcs:  # some arcs have been previously "discovered" during fragment filtering.
                        continue

                    candidate_pairs = ((f, g) for f in self.min_fragments_to[d] for g in self.min_fragments_from[p])
                    for f, g in candidate_pairs:
                        if f.end.loc != g.start.loc + self.data.n and \
                                self.is_legal((f, g), complete=True, check_cover=False):
                            self.dp_arcs.add(a)
                            break

        for a in self.dp_arcs:
            self.arcs_by_end[a.end].add(a)
            self.arcs_by_start[a.start].add(a)

        self.start_arcs = {LArc(self.o_depot, p) for p in self.p_nodes if len(p.load) == 0}
        for a in self.start_arcs:
            self.arcs_by_end[a.end].add(a)
        self.arcs_by_start[self.o_depot] = self.start_arcs

        self.end_arcs = {LArc(d, self.d_depot) for d in self.d_nodes if len(d.load) == 0}
        for a in self.end_arcs:
            self.arcs_by_start[a.start].add(a)
        self.arcs_by_end[self.d_depot] = self.end_arcs

        self.arcs = self.start_arcs | self.dp_arcs | self.end_arcs


    def network_reduce(self):
        def try_discard(d : dict, key, val):
            try:
                d[key].discard(val)
            except KeyError:
                pass

        p_nodes_to_remove = list(p for p in self.p_nodes if p not in self.arcs_by_end)
        d_nodes_to_remove = list(d for d in self.d_nodes if d not in self.arcs_by_start)
        print("Beginning network reduction")
        output = TablePrinter("Iter Fragments Nodes Arcs".split())
        it = 0
        output.print_line(it, len(self.fragments), len(self.p_nodes) + len(self.d_nodes) + 2, len(self.arcs))
        while p_nodes_to_remove or d_nodes_to_remove:
            new_p_nodes_to_remove = []
            for d in d_nodes_to_remove:
                F_remove = self.fragments_by_end.pop(d)
                assert len(F_remove) > 0
                for f in F_remove:
                    self.fragments.remove(f)
                    self.min_fragments_to[f.end].discard(f)
                    self.min_fragments_from[f.start].discard(f)
                    F_start = self.fragments_by_start[f.start]
                    F_start.remove(f)
                    if len(F_start) == 0:
                        del self.fragments_by_start[f.start]
                        new_p_nodes_to_remove.append(f.start)
                        raise Exception
                self.d_nodes.remove(d)

            new_d_nodes_to_remove = []
            for p in p_nodes_to_remove:
                F_remove = self.fragments_by_start.pop(p)
                assert len(F_remove) > 0
                for f in F_remove:
                    self.fragments.remove(f)
                    self.min_fragments_to[f.end].discard(f)
                    self.min_fragments_from[f.start].discard(f)
                    F_end = self.fragments_by_end[f.end]
                    F_end.remove(f)
                    if len(F_end) == 0:
                        del self.fragments_by_end[f.end]
                        new_d_nodes_to_remove.append(f.end)
                        raise Exception
                self.p_nodes.remove(p)

            p_nodes_to_remove = new_p_nodes_to_remove
            d_nodes_to_remove = new_d_nodes_to_remove
            it += 1
            output.print_line(it, len(self.fragments), len(self.p_nodes) + len(self.d_nodes) + 2, len(self.arcs))
        self.min_fragments_to = {k : v for k,v in self.min_fragments_to.items() if len(v) > 0}
        self.min_fragments_from = {k : v for k,v in self.min_fragments_from.items() if len(v) > 0}

    def add_arcs_and_nodes_part2(self):
        table = [
            ('Start Arcs', len(self.start_arcs)),
            ('End Arcs', len(self.end_arcs)),
            ('DP Arcs', len(self.dp_arcs)),
            ('Arcs', len(self.arcs)),
            ('Fragments', len(self.fragments)),
            ('D Nodes', len(self.d_nodes)),
            ('P Nodes', len(self.p_nodes)),
            ('Nodes', len(self.d_nodes) + len(self.p_nodes) + 2),
        ]
        output = TablePrinter(['Final network size:', ''], min_col_width=8)
        for row in table:
            output.print_line(*row)
        for k, v in table:
            k = 'final_size_' + k.lower().replace(' ', '_')
            self.network_size_info[k] = v


    def finalise_lookups(self):
        for f in self.fragments:
            for i in f.path:
                self.fragments_by_loc[i].add(f)
            self.fragments_by_locset[frozenset(f.path)].add(f)

        self.max_fragment_length = max(map(len, self.fragments_by_locset.keys()))

        for a in self.arcs:
            self.arc_cost[a] = self.data.travel_cost[a.start.loc, a.end.loc]

        for d in self.d_nodes:
            self.d_nodes_by_load[d.load].add(d)

        for p in self.p_nodes:
            self.p_nodes_by_load[p.load].add(p)

    def freeze_dicts(self):
        skip = {'network_size_info', 'legal_fragments_before_fragment_lookup', 'legal_fragments_after_fragment_lookup'}
        for attr in vars(self):
            if attr in skip or attr.startswith('_'):
                continue

            val = getattr(self, attr)
            if isinstance(val, dict):
                setattr(self, attr, frozendict(val))


    def is_legal(self, chain, check_cover=True, complete=True):
        if __debug__:
            self._is_legal_cache_info['total'] += 1

        if chain in self._is_legal_cache:
            (was_completed, legal) = self._is_legal_cache[chain]

            if not was_completed and legal and complete:
                # The chain may still be illegal due having no legal ends or starts, so now we check its completions.
                # Cover has already been checked
                legal = self._is_legal(chain, check_cover=False, complete=True)
                self._is_legal_cache[chain] = (True, legal)
        else:
            legal = self._is_legal(chain, check_cover=check_cover, complete=complete)
            if complete or (not legal):
                self._is_legal_cache[chain] = legal
            self._is_legal_cache[chain] = (complete, legal)

        return legal


    def _is_legal(self, chain: Chain, check_cover=True, complete=True):
        if __debug__:
            self._is_legal_cache_info['misses'] += 1
        if check_cover:
            locs_visited = set()
            path = ()
            for f in chain:
                for i in f.path:
                    if i in locs_visited:
                        # print('cover violation')
                        return False
                    locs_visited.add(i)
                path = path + f.path
        else:
            path = tuple(i for f in chain for i in f.path)

        if get_early_schedule(path, self.data, start_time=0, check_illegal=True) is None:
            # print('schedule violation')
            return False

        if complete:
            if not check_cover:
                locs_visited = set(i for f in chain for i in f.path)

            end_node = chain[-1].end
            start_node = chain[0].start

            end_empty = len(end_node.load) == 0
            start_empty = len(start_node.load) == 0

            if not start_empty and not end_empty:
                for a1 in self.arcs_by_end[start_node]:
                    if a1.start.loc in locs_visited:
                        continue
                    for f1 in self.min_fragments_to[a1.start]:
                        if not self.is_legal((f1,) + chain, complete=False, check_cover=False):
                            continue
                        for a2 in self.arcs_by_start[end_node]:
                            if a2.end.loc in locs_visited or a2.end.loc == a1.start.loc - self.data.n:
                                continue
                            for f2 in self.min_fragments_from[a2.end]:
                                # Chain is already completed, but setting complete=True is stronger for caching purposes
                                # if self.is_legal((f1,) + chain + (f2,), complete=False, check_cover=False):
                                if self.is_legal((f1,) + chain + (f2,), complete=True, check_cover=False):
                                    return True
                return False

            elif not start_empty and end_empty:
                for a in self.arcs_by_end[start_node]:
                    if a.start.loc in locs_visited:
                        continue
                    for f in self.min_fragments_to[a.start]:
                        # Chain is already completed, but setting complete=True is stronger for caching purposes
                        # if self.is_legal((f,) + chain, complete=False, check_cover= False):
                        if self.is_legal((f,) + chain, complete=True, check_cover= False):
                            return True
                return False

            elif start_empty and not end_empty:
                for a in self.arcs_by_start[end_node]:
                    if a.end.loc in locs_visited:
                        continue
                    for f in self.min_fragments_from[a.end]:
                        # Chain is already completed, but setting complete=True is stronger for caching purposes
                        # if self.is_legal(chain + (f,), complete=False, check_cover=False):
                        if self.is_legal(chain + (f,), complete=True, check_cover=False):
                            return True
                return False


        return True

    def legal_fragments_after_chain(self, chain: Chain, last_arc: LArc):
        legal = set()
        for f in self.fragments_by_start[last_arc.end]:
            if self.restriction is not None and f not in self.restriction.fragments:
                continue
            if self.is_legal(chain + (f,)):
                legal.add(f)
        return legal

    def legal_arcs_after_chain(self, chain: Chain, last_arc: LArc = None):
        legal = set()
        locs_visited = set(i for f in chain for i in f.path)
        for a in self.arcs_by_start[chain[-1].end]:
            if a == last_arc or a.end.loc in locs_visited:
                continue
            elif self.restriction is not None and a not in self.restriction.arcs:
                continue
            elif a.end == self.d_depot:
                legal.add(a)
            else:
                for f in self.min_fragments_from[a.end]:
                    if self.is_legal(chain + (f,)):
                        legal.add(a)
                        break
        return legal

    def legal_fragments_before_chain(self, chain: Chain, first_arc: LArc):
        legal = set()
        for f in self.fragments_by_end[first_arc.start]:
            if self.restriction is not None and f not in self.restriction.fragments:
                continue
            if self.is_legal((f,) + chain):
                legal.add(f)
        return legal

    def legal_arcs_before_chain(self, chain: Chain, first_arc: LArc = None):
        legal = set()
        locs_visited = set(i for f in chain for i in f.path)
        for a in self.arcs_by_end[chain[0].start]:
            if a == first_arc or a.start.loc in locs_visited:
                continue
            elif self.restriction is not None and a not in self.restriction.arcs:
                continue
            elif a.start == self.o_depot:
                legal.add(a)
            else:
                for f in self.fragments_by_end[a.start]:
                    if self.is_legal((f,) + chain):
                        legal.add(a)
                        break
        return legal

    @memoise
    def legal_fragments_before_arc(self, a: LArc):
        legal = set()
        assert self.restriction is None
        for f in self.fragments_by_end[a.start] - self.fragments_by_loc[a.end.loc]:
            for g in self.min_fragments_from[a.end]:
                if self.is_legal((f, g)):
                    legal.add(f)
                    break
        return legal

    @memoise
    def legal_fragments_after_arc(self, a: LArc):
        legal = set()
        assert self.restriction is None
        for f in self.fragments_by_start[a.end] - self.fragments_by_loc[a.start.loc]:
            for g in self.min_fragments_to[a.start]:
                if self.is_legal((g, f)):
                    legal.add(f)
                    break
        return legal


@dataclasses.dataclass(frozen=True)
class DARP_RF_Model_Params:
    sdarp : bool = False
    ns_heuristic : bool = False
    fr_heuristic : bool = False
    heuristics : bool = dataclasses.field(init=False)
    triad_cuts : bool = False
    AF_cuts : bool = False
    FF_cuts : bool = False
    FA_cuts : bool = False
    cuts : bool = dataclasses.field(init=False)
    cut_violate : float = 0.01
    heuristic_stop_gap : float = 0.01
    apriori_2cycle : bool = False

    def __post_init__(self):
        super().__setattr__('heuristics',self.ns_heuristic | self.fr_heuristic)
        super().__setattr__('cuts', self.AF_cuts | self.FF_cuts | self.FA_cuts | self.triad_cuts)

class DARP_RF_Model(BaseModel):
    X: BinVarDict
    Y: BinVarDict
    W: BinVarDict
    Z: IntVar
    C: CtsVar
    Xv: dict
    Yv: dict
    Zv: float
    Cv: float
    Wv: dict

    heuristic_solution: Union[None, Tuple[float, List[ResFrag]]]
    is_restricted: bool
    lc_best_obj: float
    parameters : DARP_RF_Model_Params
    _IDX = 0

    def __init__(self, network: Network, parameters : DARP_RF_Model_Params = None, cpus=None):
        super().__init__(cpus=cpus)
        if parameters is None:
            parameters = DARP_RF_Model_Params()
        self.network = network
        self.heuristic_solution = None
        self.parameters = parameters
        self.lc_best_obj = float('inf')
        self.nonzero_fragments = None
        self._gurobi_name_map = None
        self.log = {
            'nsh_succeed' : 0,
            'nsh_fail' : 0,
            'frh_succeed' : 0,
            'frh_fail' : 0,
        }
        self.model_id = self.__class__._IDX
        self.__class__._IDX += 1

        if network.restriction is not None:
            assert not self.parameters.heuristics, "error: nested heuristic models"
            F = network.restriction.fragments
            A = network.restriction.arcs
            Np = network.restriction.p_nodes
            Nd = network.restriction.d_nodes
        else:
            F = network.fragments
            A = network.arcs
            Np = network.p_nodes
            Nd = network.d_nodes

        X = {f: self.addVar(vtype=GRB.BINARY) for f in F}
        Y = {a: self.addVar(vtype=GRB.BINARY) for a in A}
        W = {p: self.addVar(vtype=GRB.BINARY) for p in network.data.P}
        Z = self.addVar(ub=len(network.data.K))
        C = self.addVar()
        self.setAttr('ModelSense', GRB.MINIMIZE)

        self.set_vars_attrs(X=X, Y=Y, Z=Z, C=C, W=W)

        self.setObjective(C)

        if self.parameters.sdarp:
            self.cons['obj'] = self.addConstr(C == self.network.data.n - quicksum(W[p] for p in self.network.data.P))
            self.cons['cover'] = {i:
                self.addLConstr(
                    quicksum(X[f] for f in self.network.fragments_by_loc[i] & F) == W[i])
                for i in self.network.data.P
            }
        else:
            self.cons['obj'] = self.addConstr(C == quicksum(self.network.path_cost[f.path] * X[f] for f in F) +
                                              quicksum(self.network.arc_cost[a] * Y[a] for a in A))

            self.cons['cover'] = {i:
                self.addLConstr(
                    quicksum(X[f] for f in self.network.fragments_by_loc[i] & F) == 1)
                for i in self.network.data.P
            }

        self.cons['flow'] = {}
        self.cons['flow'].update({d:
            self.addLConstr(
                quicksum(X[f] for f in self.network.fragments_by_end[d] & F)
                ==
                quicksum(Y[a] for a in self.network.arcs_by_start[d] & A)
            )
            for d in Nd
        })

        self.cons['flow'].update({p:
            self.addLConstr(
                quicksum(X[f] for f in self.network.fragments_by_start[p] & F)
                ==
                quicksum(Y[a] for a in self.network.arcs_by_end[p] & A)
            )
            for p in Np
        })

        self.cons['num_vehicles'] = self.addLConstr(quicksum(Y[a] for a in self.network.start_arcs & A) == Z)

        if network.restriction is None and parameters.apriori_2cycle:
            self.cons['2cycle'] = {}
            for a in network.dp_arcs:
                if len(network.fragments_by_start[a.end] & network.fragments_by_end[a.start]) > 0:
                    self.cons['2cycle'][a,'after'] = self.addLConstr(Y[a] <= quicksum(X[f] for f in network.legal_fragments_after_arc(a)))
                    self.cons['2cycle'][a,'before'] = self.addLConstr(Y[a] <= quicksum(X[f] for f in network.legal_fragments_before_arc(a)))
                    # print(str(a))


    def translate_gurobi_name(self, name: str):
        if self._gurobi_name_map is None:
            self._gurobi_name_map = {}
            for f, var in self.X.items():
                self._gurobi_name_map[var.VarName] = f
            for a, var in self.Y.items():
                self._gurobi_name_map[var.VarName] = a

        return self._gurobi_name_map[name]

    def optimize(self, callback=None):
        super().optimize(callback)


def build_chains(model: DARP_RF_Model):
    frags = defaultdict(dict)
    arcs = defaultdict(dict)

    for f, val in model.Xv.items():
        frags[f.start][f] = val

    for a, val in model.Yv.items():
        arcs[a.start][a] = val

    frags.default_factory = None
    arcs.default_factory = None

    chains = []
    cycles = []

    while len(frags) > 0:
        a = take(arcs.get(model.network.o_depot, ()))

        if a is not None:
            is_cycle = False
            val = arcs[model.network.o_depot][a]
        else:
            is_cycle = True
            a, val = take(arcs[take(arcs)].items())

        edges = [a]
        visited_nodes = {a.start: 0, a.end: 1}
        node = a.end

        while True:
            f, vf = take(frags[node].items())
            node = f.end
            edges.append(f)
            val = min(val, vf)

            if node in visited_nodes:
                is_cycle = True
                k = visited_nodes[node]
                edges = edges[k:]
                assert isinstance(edges[0], LArc)
                break

            visited_nodes[node] = len(visited_nodes)

            a, va = take(arcs[node].items())
            node = a.end
            edges.append(a)
            val = min(val, va)

            if node == model.network.d_depot:
                break
            elif node in visited_nodes:
                is_cycle = True
                k = visited_nodes[node]
                edges = edges[k:]
                assert isinstance(edges[0], ResFrag)
                # cycle shift an arc to the start
                edges = [edges[-1]] + edges[:-1]
                assert isinstance(edges[0], LArc)
                break

            visited_nodes[node] = len(visited_nodes)

        c_arcs = edges[::2]
        c_frags = edges[1::2]

        if is_cycle:
            val = min(itertools.chain((arcs[a.start][a] for a in c_arcs), (frags[f.start][f] for f in c_frags)))

        for c_edge, d in zip([c_frags, c_arcs], [frags, arcs]):
            for e in c_edge:
                remaining = d[e.start][e] - val
                if abs(remaining) < EPS:
                    del d[e.start][e]
                    if len(d[e.start]) == 0:
                        del d[e.start]
                else:
                    d[e.start][e] = remaining

        c_frags = tuple(c_frags)
        c_arcs = tuple(c_arcs)

        if is_cycle:
            assert len(c_arcs) == len(c_frags)
            cycles.append((c_frags, c_arcs, val))
        else:
            assert len(c_arcs) == len(c_frags) + 1
            chains.append((c_frags, c_arcs, val))

    return chains, cycles


def separate_legality_cuts(model: DARP_RF_Model):
    cuts = {}
    chains, cycles = build_chains(model)
    legal_routes = []
    illegal_routes = []

    for fragments, arcs, _ in chains:

        if model.network.is_legal(fragments, check_cover=False):
            legal_routes.append(fragments)
            continue

        for chain_end in range(1, len(fragments)):
            if not model.network.is_legal(fragments[:chain_end + 1], check_cover=False):
                break
        else:
            raise Exception

        for chain_start in reversed(range(chain_end)):
            if not model.network.is_legal(fragments[chain_start:chain_end + 1], check_cover=False):
                break
        else:
            raise Exception

        illegal_routes.append(fragments)
        illegal_chain = fragments[chain_start:chain_end + 1]
        illegal_arcs = arcs[chain_start + 1:chain_end + 1]
        # print(f'illegal chain {_IDX:d}'.center(80, '-'))
        # print('F:', *map(str, illegal_chain))
        # print('A:', *map(str, illegal_arcs))
        pth = tuple(i for f in illegal_chain for i in f.path)
        # pprint_path(pth, model.network.data)
        # print(*(f"{model.network.data.travel_time[i, j]:.3f}" for i, j in zip(pth, pth[1:])))
        # print(''.center(80, '-'))
        F_after = model.network.legal_fragments_after_chain(illegal_chain[:-1], illegal_arcs[-1])
        if len(F_after) > 0:

            after_cut = (
                    quicksum(model.X[f] for f in illegal_chain[:-1]) +
                    quicksum(model.Y[a] for a in illegal_arcs) <=
                    quicksum(model.X[f] for f in F_after) +
                    len(illegal_chain) + len(illegal_arcs) - 2
            )
            cuts[illegal_chain, 'after'] = after_cut

        else:
            A_after = model.network.legal_arcs_after_chain(illegal_chain[:-1], illegal_arcs[-1])
            # assert len(A_after) > 0
            after_cut = (
                    quicksum(model.X[f] for f in illegal_chain[:-1]) +
                    quicksum(model.Y[a] for a in illegal_arcs[:-1]) <=
                    quicksum(model.Y[a] for a in A_after) +
                    len(illegal_chain) + len(illegal_arcs) - 3
            )

            cuts[illegal_chain, 'after'] = after_cut

        F_before = model.network.legal_fragments_before_chain(illegal_chain[1:], illegal_arcs[0])
        if len(F_before) > 0:  # or len(illegal_chain) == 2:
            before_cut = (
                    quicksum(model.X[f] for f in illegal_chain[1:]) +
                    quicksum(model.Y[a] for a in illegal_arcs) <=
                    quicksum(model.X[f] for f in F_before) +
                    len(illegal_chain) + len(illegal_arcs) - 2
            )

            cuts[illegal_chain, 'before'] = before_cut


        else:
            A_before = model.network.legal_arcs_before_chain(illegal_chain[1:], illegal_arcs[0])
            # assert len(A_before) > 0 # this condition may not hold for restricted models
            before_cut = (
                    quicksum(model.X[f] for f in illegal_chain[1:]) +
                    quicksum(model.Y[a] for a in illegal_arcs[1:]) <=
                    quicksum(model.Y[a] for a in A_before) +
                    len(illegal_chain) + len(illegal_arcs) - 3
            )

            cuts[illegal_chain, 'before'] = before_cut

    for frags, arcs, _ in cycles:
        # print('cycle'.center(80, '-'))
        # print(*map(str, frags))
        cut = (
                quicksum(model.X[f] for f in frags) + quicksum(model.Y[a] for a in arcs)
                <= len(frags) + len(arcs) - 1
        )
        cuts[frags, 'cycle'] = cut

    return cuts, legal_routes, illegal_routes, cycles


@cut('triad_cuts', True)
def separate_triad_cuts(model: DARP_RF_Model, violate):
    pairs = defaultdict(lambda: 0.0)
    triples = defaultdict(lambda: 0.0)
    for f, val in model.Xv.items():
        for p in model.network.path_loc_pairs(f.path):
            pairs[p] += val
        for t in model.network.path_loc_triples(f.path):
            triples[t] += val

    cuts = dict()
    checked = set()
    for pair1, pair2 in itertools.combinations(list(pairs.keys()), 2):
        triple = pair1 | pair2
        if len(triple) == 3 and triple not in checked:
            checked.add(triple)
            if True or triple not in model.cut_cache.get('triad_cuts', {}):
                cut_value = pairs[pair1] + pairs[pair2] + pairs[pair1 ^ pair2] - 2 * triples[triple]
                if cut_value > 1 + violate:
                    cuts[frozenset(triple)] = (
                            quicksum(model.X[f] for f in model.network.get_triad(triple)) <= 1
                    )

    return cuts


@cut('FA_cuts', False)
def separate_fragment_arc_cuts(model: DARP_RF_Model, violate):
    cuts = {}
    for f, lhs in model.Xv.items():
        if lhs > violate:
            cache_key = (f,'after')
            A = model.network.legal_arcs_after_chain((f,))
            rhs = sum(model.Yv.get(a, 0) for a in A)
            if lhs > rhs + violate:
                cuts[cache_key] = model.X[f] <= quicksum(model.Y[a] for a in A)

            cache_key = (f,'before')
            A = model.network.legal_arcs_before_chain((f,))
            rhs = sum(model.Yv.get(a, 0) for a in A)
            if lhs > rhs + violate:
                cuts[f, 'before'] = model.X[f] <= quicksum(model.Y[a] for a in A)

    return cuts


@cut('AF_cuts', False)
def separate_arc_fragment_cuts(model: DARP_RF_Model, violate):
    cuts = {}
    for a, lhs in model.Yv.items():
        if lhs > violate and a in model.network.dp_arcs:
            cache_key = (a, 'before')
            if True or cache_key not in model.cut_cache.get('AF_cuts', {}):
                F = model.network.legal_fragments_before_arc(a)
                rhs = sum(model.Xv.get(f, 0) for f in F)
                if lhs > rhs + violate:
                    cuts[cache_key] = (model.Y[a] <= quicksum(model.X[f] for f in F))

            cache_key = (a, 'after')
            if True or cache_key not in model.cut_cache.get('AF_cuts', {}):
                F = model.network.legal_fragments_after_arc(a)
                rhs = sum(model.Xv.get(f, 0) for f in F)
                if lhs > rhs + violate:
                    cuts[cache_key] = (model.Y[a] <= quicksum(model.X[f] for f in F))
    return cuts


@cut('FF_cuts', False)
def separate_fragment_fragment_cuts(model: DARP_RF_Model, violate):
    cuts = {}
    for f, lhs in model.Xv.items():
        if lhs > violate:
            if len(f.end.load) > 0:
                if f in model.network.legal_fragments_after_fragment_lookup:
                    F = model.network.legal_fragments_after_fragment_lookup[f]
                    rhs = sum(model.Xv.get(g,0) for g in F)
                    addcut = (lhs > rhs + violate)
                else:
                    F = set()
                    rhs = 0
                    candidates = (g for a in model.network.arcs_by_start[f.end]
                                  for g in model.network.fragments_by_start[a.end])
                    for g in candidates:
                        if model.network.is_legal((f, g)):
                            F.add(g)
                            rhs += model.Xv.get(g, 0)
                            if lhs <= rhs:
                                addcut = False
                                break
                    else:
                        model.network.legal_fragments_after_fragment_lookup[f] = F
                        addcut = (lhs > rhs + violate)

                if addcut:
                    cuts[f, 'after'] = (model.X[f] <= quicksum(model.X[fs] for fs in F))


            if len(f.start.load) > 0:
                if f in model.network.legal_fragments_before_fragment_lookup:
                    F = model.network.legal_fragments_before_fragment_lookup[f]
                    rhs = sum(model.Xv.get(g, 0) for g in F)
                    addcut = (lhs > rhs + violate)
                else:
                    F = set()
                    rhs = 0
                    candidates = (g for a in model.network.arcs_by_end[f.start]
                                  for g in model.network.fragments_by_end[a.start])
                    for g in candidates:
                        if model.network.is_legal((g, f)):
                            F.add(g)
                            rhs += model.Xv.get(g, 0)
                            if lhs <= rhs:
                                addcut = False
                                break
                    else:
                        addcut = (lhs > rhs + violate)

                if addcut:
                    cuts[f,'before'] = (model.X[f] <= quicksum(model.X[fs] for fs in F))

    return cuts


def add_valid_inequalities(model: DARP_RF_Model, where, cutdict, name):
    if where is None:
        if name not in model.cons:
            model.cons[name] = dict()
        for key, cut in cutdict.items():
            assert key not in model.cons[name]
            model.cons[name][key] = model.addConstr(cut)

    elif where == GRB.Callback.MIPNODE:
        for key, cut in cutdict.items():
            model.cbCut(cut, name, key)

    else:
        raise NotImplementedError

def add_lazy_cuts(model: DARP_RF_Model, where, cutdict, name):
    if where is None:
        if name not in model.cons:
            model.cons[name] = dict()
        for key, cut in cutdict.items():
            assert key not in model.cons[name]
            model.cons[name][key] = model.addConstr(cut)

    elif where == GRB.Callback.MIPNODE or where == GRB.Callback.MIPSOL:
        for key, cut in cutdict.items():
            model.cbLazy(cut, name, key)

    else:
        raise NotImplementedError


def main_callback(model: DARP_RF_Model, where):
    if where == GRB.Callback.MIPSOL:
        model.update_var_values(where, eps=.9)
        cuts, legal_routes, illegal_routes, cycles = separate_legality_cuts(model)
        if len(cuts) > 0:
            add_lazy_cuts(model, where, cuts, 'legality')
        else:  # legal solution
            # Keep track of this manually - weird gurobi behaviour around MIPSOL_OBJBST and cutting off solutions.
            model.lc_best_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if model.parameters.heuristics and model.heuristic_solution is None:
            current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            ub = model.lc_best_obj
            lb = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            n_legal_routes = len(legal_routes)
            n_illegal_routes = len(illegal_routes)
            n_cycles = len(cycles)
            if (ub - lb) / (lb + EPS) > model.parameters.heuristic_stop_gap:
                heuristic_succeeded = False
                print(' HEURISTICS '.center(80, '-'))
                print(f'{n_legal_routes} legal routes, {n_illegal_routes} illegal routes and {n_cycles} cycles.')
                if model.parameters.ns_heuristic:  # lel, need to check this, otherwise we recursively create models forever.
                    print('beginning Neighbourhood Search Heuristic')
                    s = Stopwatch().start()
                    new_ub, frags, arcs = neighbourhood_search_heuristic(model, ub)
                    if new_ub < float('inf'):
                        if new_ub + EPS < ub:
                            print(f'NSH found solution: {ub:.3f} -> {new_ub:.3f}')
                            model.heuristic_solution = (new_ub, frags, arcs)
                            heuristic_succeeded = True
                            model.log['nsh_succeed'] += 1
                        else:
                            print(f'NSH failed to improve solution: {new_ub:.3f}')
                            model.log['nsh_fail'] += 1
                    else:
                        print('NSH was infeasible.')
                        model.log['nsh_fail'] += 1
                    s.stop()
                    print(f'NSH took {s.time:.2f}s ')

                if (model.parameters.fr_heuristic and
                        not heuristic_succeeded and
                        n_illegal_routes > 0
                ):
                    legal_route_locs = sum(len(f.path) for r in legal_routes for f in r)
                    legal_route_cost = 0
                    for r in legal_routes:
                        p = [i for f in r for i in f.path]
                        legal_route_cost += sum(model.network.data.travel_cost[i, j] for i, j in zip(p, p[1:]))

                    if legal_route_cost < ub * (legal_route_locs / (2 * model.network.data.n)):
                        print('beginning Fixed-Route Heuristic')
                        s = Stopwatch().start()
                        new_ub, frags, arcs = fixed_route_heuristic(model, ub, legal_routes)
                        if new_ub < float('inf'):
                            if new_ub + EPS < ub:
                                print(f'FRH found solution: {ub:.3f} -> {new_ub:.3f}')
                                model.heuristic_solution = (new_ub, frags, arcs)
                                heuristic_succeeded = False
                                model.log['frh_succeed'] += 1
                            else:
                                print(f'FRH failed to improve solution: {new_ub:.3f}')
                                model.log['frh_fail'] += 1
                        else:
                            print('FRH was infeasible.')
                            model.log['frh_fail'] += 1
                        s.stop()
                        print(f'FRH took {s.time:.2f}s')

                print('-' * 80)


    elif where == GRB.Callback.MIPNODE:
        ub = model.lc_best_obj

        if model.heuristic_solution is not None:
            heur_ub, frags, arcs = model.heuristic_solution

            if heur_ub + EPS < ub:
                print(f'post solution: {heur_ub:.3f} ', end='')
                model.cbSetSolution([model.X[f] for f in frags] + [model.Y[a] for a in arcs],
                                    [1] * (len(frags) + len(arcs)))
                obj = model.cbUseSolution()
                print(f'({obj:.3f})')
                assert math.isclose(obj, heur_ub, rel_tol=1e-6)
                ub = heur_ub
            else:
                model.heuristic_solution = None

        elif model.parameters.cuts:
            cut_stop_gap = 0.001
            cut_local_gap = 0.0025
            cut_max_nodes = 10
            if (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL) and model.network.restriction is None:
                lb = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                global_gap = 1 - lb/(1e-8 + ub)
                if global_gap > cut_stop_gap:
                    node_lb = model.cbGetNodeRel(model.C)
                    local_gap = 1 - lb / (node_lb + 1e-8)
                    # print(f'CUTS: {ub:.3f} {node_lb:.3f} {lb:.3f} {local_gap*100:.3f}% {global_gap*100:.3f}%')

                    # if ((local_gap < cut_local_gap and local_gap < 0.1*global_gap)
                    #         or model.cbGet(GRB.Callback.MIPNODE_NODCNT) <= cut_max_nodes):
                    if model.cbGet(GRB.Callback.MIPNODE_NODCNT) <= cut_max_nodes:
                        model.update_var_values(where)
                        for cutname, sep_func in CUT_FUNC_LOOKUP.items():
                            if getattr(model.parameters, cutname):
                                cuts = sep_func(model, model.parameters.cut_violate)

                                if len(cuts) > 0:
                                    print(cutname.rjust(20) + ":", len(cuts))
                                    if cutname in VALID_INEQUALITIES:
                                        add_valid_inequalities(model, where, cuts, cutname)
                                    else:
                                        add_lazy_cuts(model, where, cuts, cutname)
                    # else:
                    #     print('(skipping cuts)')

_neighbourhood_search_heuristic_cache = dict()
_neighbourhood_search_heuristic_cache_hits = 0
def neighbourhood_search_heuristic(model: DARP_RF_Model, ub):
    locsets = set(map(lambda f: frozenset(f.path), model.Xv.keys()))
    # merge all locsets of length 2 as well
    for a, b in itertools.combinations(filter(lambda x: len(x) == 2, locsets), 2):
        locsets.add(a | b)

    fragments = set()
    for s in locsets:
        fragments |= model.network.get_fragments_by_coverset(s)

    if model.nonzero_fragments is not None:
        fragments &= model.nonzero_fragments

    fragments = frozenset(fragments)

    print(f'restricted to {len(fragments)} fragments.')

    if len(fragments) > 1500:
        print('too many fragments, giving up.')
        return float('inf'), [], []

    cache_key = (model.model_id, fragments)
    if cache_key in _neighbourhood_search_heuristic_cache:
        if __debug__:
            global _neighbourhood_search_heuristic_cache_hits
            _neighbourhood_search_heuristic_cache_hits += 1
        return _neighbourhood_search_heuristic_cache[cache_key]

    with model.network.restrict(fragments):
        restricted_model = DARP_RF_Model(model.network, cpus=model.cpus)

        for param, value in HEURISTIC_MODEL_PARAMETERS.items():
            restricted_model.setParam(param, value)

        restricted_model.setParam('BestBdStop', ub)
        restricted_model.Z.BranchPriority = 10
        restricted_model.optimize(main_callback)

    if restricted_model.SolCount > 0:
        restricted_model.update_var_values()
        retval = (restricted_model.ObjVal, list(restricted_model.Xv.keys()), list(restricted_model.Yv.keys()))
    else:
        retval = (float('inf'), [], [])

    _neighbourhood_search_heuristic_cache[cache_key] = retval
    return retval

_fixed_route_heuristic_cache = {}
_fixed_route_heuristic_cache_hits = 0
def fixed_route_heuristic(model: DARP_RF_Model, ub, legal_routes):
    fixed_fragments = set(f for r in legal_routes for f in r)
    fixed_locations = set(i for f in fixed_fragments for i in f.path)

    unfixed_locations = frozenset(set(range(1, 2 * model.network.data.n + 1)) - fixed_locations)
    unfixed_fragments = set()
    for locset, frags in model.network.fragments_by_locset.items():
        if locset <= unfixed_locations:
            unfixed_fragments.update(frags)

    if model.nonzero_fragments is not None:
        unfixed_fragments &= model.nonzero_fragments
        fixed_fragments &= model.nonzero_fragments

    F = frozenset(unfixed_fragments | fixed_fragments)

    print(f'restricted to {len(F)} fragments')

    if len(F) > 1500:
        print('too many fragments, giving up.')
        return float('inf'), [], []

    cache_key = (model.model_id, F)
    if cache_key in _neighbourhood_search_heuristic_cache:
        if __debug__:
            global _neighbourhood_search_heuristic_cache_hits
            _neighbourhood_search_heuristic_cache_hits += 1
        return _neighbourhood_search_heuristic_cache[cache_key]

    with model.network.restrict(F):
        restricted_model = DARP_RF_Model(model.network, cpus=model.cpus)

        for f in fixed_fragments:
            restricted_model.X[f].lb = 1

        restricted_model.Z.BranchPriority = 10

        for param, value in HEURISTIC_MODEL_PARAMETERS.items():
            restricted_model.setParam(param, value)

        restricted_model.setParam('BestBdStop', ub)
        restricted_model.optimize(main_callback)

    if restricted_model.SolCount > 0:
        restricted_model.update_var_values()
        retval = (restricted_model.ObjVal, list(restricted_model.Xv.keys()), list(restricted_model.Yv.keys()))
    else:
        retval = (float('inf'), [], [])

    _fixed_route_heuristic_cache[cache_key] = retval
    return retval

class DARPRestrictedFragmentsExperiment(DARP_Experiment):
    ROOT_PATH = DARP_Experiment.ROOT_PATH / 'rf'
    INPUTS = {**{
        "sdarp": {"type": "boolean", "default": False, "help": "Solve the Selective-DARP instead.  Affects INDEX --"
                                                               " Riedler (2018) dataset is used instead"},
    }, **DARP_Experiment.INPUTS}
    PARAMETERS = {**{
        "cut_violate": {"type": "float", "min": 0, "max": 1, "default": 0.01},
        "filter_rf": {"type": "boolean", "default": True},
        "triad_cuts": {"type": "boolean", "default": True},
        "FA_cuts": {"type": "boolean", "default": True},
        "AF_cuts": {"type": "boolean", "default": True},
        "FF_cuts": {"type": "boolean", "default": True},
        "domination": {"type": "boolean", "default": True},
        "NS_heuristic": {"type": "boolean", "default": True},
        "FR_heuristic": {"type": "boolean", "default": True},
        "ap2c" : {"type" : "boolean", "default" : False, "help": "ap2c stands for A Priori 2-Cycle constraints"},
        "heuristic_stop_gap": {"type": "float", "default": 0.01, "min": 0, "max": 1},
        "rc_frac": {"type": "float", "default": 0.1, "max": 1,
                    "help": "Fraction of fragments used in the Reduced-Cost heuristic model. Setting this to a "
                            "negative number disables this heuristic."},
        "rc_fix" : {"type": "boolean", "default": True,
                    "help" : "Fixes variables to 0 based on RC and UB prior to main MIP.  Only happens if RC_FRAC is "
                             "non-negative"},
    }, **DARP_Experiment.PARAMETERS}

    def define_derived(self):
        if self.inputs['sdarp']:
            self.inputs["instance"] = get_name_by_index("sdarp", self.inputs["index"])
        else:
            self.inputs["instance"] = get_name_by_index("darp", self.inputs["index"])
        self.outputs["info"] = self.get_output_path("info.json")
        self.outputs["solution"] = self.get_output_path("soln.json")
        super(DARP_Experiment, self).define_derived()

    @property
    def input_string(self):
        s = "RF" + super().input_string
        if self.inputs['sdarp']:
            s = "SDARP_" + s
        return s

    def _get_time_and_memory(self, idx, extend, ridetime):
        if idx in NAMED_DATASET_INDICES['a'] | NAMED_DATASET_INDICES['b']:
            if extend < 3:
                return '5GB', 300

            if idx in NAMED_DATASET_INDICES['medium']:
                return '5GB', 300

            if idx < 47:
                return '10GB', 600

        return '50GB', self.parameters['timelimit'] + 300


    @property
    def resource_time(self) -> str:
        _, t = self._get_time_and_memory(self.inputs['index'], self.inputs['extend'], self.inputs['ridetime'])
        t_max = self.parameters['timelimit'] + 300
        if not self.parameters['triad_cuts']:
            t *= 2
        if not self.parameters['NS_heuristic']:
            t*= 2
        t = min(t, t_max)
        return slurm_format_time(t)

    @property
    def resource_memory(self) -> str:
        mem, _ = self._get_time_and_memory(self.inputs['index'], self.inputs['extend'], self.inputs['ridetime'])
        return mem

    @property
    def resource_name(self) -> str:
        return f'{self.parameter_string}/{self.input_string}'


def write_info_file(exp : DARP_Experiment, times,
                    network : Network, obj_info, cut_counter=None, model : DARP_RF_Model = None):

    if model is None:
        info = {}
    else:
        info = model.get_gurobi_model_information()
        info = dataclasses.asdict(info)
        info['mip_constraints'] = model.cons_size
        info['log'] = model.log

    info['network_size'] = network.network_size_info
    info['time'] = dict(times)
    info['bounds'] = obj_info
    if cut_counter is not None:
        info['cuts'] = dict(cut_counter)

    with open(exp.outputs['info'], 'w') as fp:
        json.dump(info, fp, indent='  ')

if __name__ == '__main__':
    stopwatch = Stopwatch()
    exp = DARPRestrictedFragmentsExperiment.from_cl_args()

    exp.print_summary_table()
    exp.write_index_file()
    data = exp.data

    # noinspection PyArgumentList
    model_custom_params = DARP_RF_Model_Params(
        sdarp=exp.inputs['sdarp'],
        ns_heuristic=exp.parameters['NS_heuristic'],
        fr_heuristic=exp.parameters['FR_heuristic'],
        heuristic_stop_gap=exp.parameters['heuristic_stop_gap'],
        triad_cuts=exp.parameters['triad_cuts'],
        AF_cuts=exp.parameters['AF_cuts'],
        FF_cuts=exp.parameters['FF_cuts'],
        FA_cuts=exp.parameters['FA_cuts'],
        cut_violate=exp.parameters['cut_violate'],
        apriori_2cycle=exp.parameters['ap2c']
    )

    stopwatch.start()
    data = tighten_time_windows(data)
    data = remove_arcs(data)
    stopwatch.lap('data_preprocess')
    network = Network(data,
                      fragment_domination=exp.parameters['domination'],
                      filter_fragments=exp.parameters['filter_rf'])
    network.build()
    stopwatch.lap('network_build')
    obj_info = {}
    write_info_file(exp, stopwatch.times.copy(),network, obj_info)

    model = DARP_RF_Model(network, model_custom_params, cpus=exp.parameters['cpus'])
    stopwatch.lap('model_build')
    model.set_variables_continuous()
    model.optimize()
    obj_info['lb_root_lp'] = model.ObjVal
    cut_counter = defaultdict(int)

    write_info_file(exp, stopwatch.times.copy(),network, obj_info,model=model)

    num_cuts_added = float('inf')
    output = TablePrinter(list(CUT_FUNC_LOOKUP.keys()) + ['objective'])

    with model.temp_params(OutputFlag=0):
        while True:
            if num_cuts_added == float('inf'):  # first iteration only.
                output.print_line(model.ObjVal, pad_left=True)
            else:
                model.optimize()

            num_cuts_added = 0
            model.update_var_values()
            cuts_this_iter = []
            for cutname in CUT_FUNC_LOOKUP:
                if getattr(model.parameters, cutname):
                    separate_func = CUT_FUNC_LOOKUP[cutname]
                    cuts = separate_func(model, violate=model.parameters.cut_violate)
                    if cutname in VALID_INEQUALITIES:
                        add_valid_inequalities(model, None, cuts, cutname)
                    else:
                        add_lazy_cuts(model, None, cuts, cutname)
                    ncuts = len(cuts)
                    cut_counter[cutname + '_lp'] += ncuts
                    num_cuts_added += ncuts
                else:
                    ncuts = 0
                cuts_this_iter.append(ncuts)

            if num_cuts_added == 0:
                break

            output.print_line(*[x if x > 0 else '' for x in cuts_this_iter], model.ObjVal)

    obj_info['lb_lifted_lp'] = model.ObjVal
    stopwatch.lap('rootcut')

    write_info_file(exp, stopwatch.times.copy(),network, obj_info,cut_counter, model)

    # Reduced-cost heuristic
    for param, value in MASTER_MODEL_PARAMETERS.items():
        model.setParam(param, value)
    for param, value in exp.parameters['gurobi'].items():
        model.setParam(param, value)

    if model.parameters.sdarp:
        for w in model.W.values():
            w.BranchPriority = 10
    else:
        model.Z.BranchPriority = 10

    if exp.parameters['rc_frac'] > 0:
        reduced_costs = sorted([(var.RC, f) for f, var in model.X.items()], key=lambda x: x[0])
        reduced_costs_cutoff_idx = int(exp.parameters['rc_frac'] * len(reduced_costs))
        rc_heur = model.addConstr(quicksum(model.X[f] for _, f in reduced_costs[reduced_costs_cutoff_idx:]) == 0)

        model.nonzero_fragments = set(f for _, f in reduced_costs[:reduced_costs_cutoff_idx])
        model.set_variables_integer()
        model.update()
        with model.temp_params(TimeLimit=(exp.parameters['timelimit']-stopwatch.time)/2, MIPGap=0.01):
            model.optimize(main_callback)

        # noinspection PyUnreachableCode
        if __debug__:
            if model.Status == GRB.OPTIMAL:
                model.update_var_values()
                rc_heur_soln = {'X': model.Xv.copy(), 'Y': model.Yv.copy()}
                rc_heur_chains, _ = build_chains(model)
                for F, _, _ in rc_heur_chains:
                    pprint_path(tuple(i for f in F for i in f.path), data)

        model.remove(rc_heur)
        model.nonzero_fragments = None
        print(f'Adding {model.cut_cache_size:d} constraints as hard constraints.')
        for cutname, cutdict in model.cut_cache.items():
            cut_counter[cutname + '_rc_mip'] = len(cutdict)


        ub_rc_mip = model.lc_best_obj

        model.flush_cut_cache()
        if exp.parameters['rc_fix'] and ub_rc_mip < float('inf'):
            model.set_variables_continuous()
            with model.temp_params(OutputFlag=0):
                model.optimize()
            gap = ub_rc_mip - model.ObjVal
            n_fixed = 0
            for var in model.X.values():
                if var.rc > gap + 0.01:
                    var.ub = 0
                    n_fixed += 1
            print(f"RC-fixing: fixed {n_fixed} variables")
            model.set_variables_integer()

        obj_info['ub_rc_mip'] = model.lc_best_obj
    else:
        model.set_variables_integer()

    stopwatch.lap('rc_mip')

    write_info_file(exp, stopwatch.times.copy(),network, obj_info,cut_counter, model)
    if __debug__:
        model.set_variables_continuous()
        model.optimize()
        model.update_var_values()
        chains, cycles = build_chains(model)
        for F, A, val in chains:
            print(f"{val:.5f} " + " ".join(map(lambda x : f"{str(x):30s}", F)))
        model.set_variables_integer()

    with model.temp_params(TimeLimit=exp.parameters['timelimit']-stopwatch.time):
        model.optimize(main_callback)
    stopwatch.stop('main_mip')

    for cutname, cutdict in model.cut_cache.items():
        cut_counter[cutname + '_main_mip'] = len(cutdict)

    obj_info['ub_final'] = model.ObjVal
    obj_info['lb_final'] = model.ObjBound

    if model.SolCount > 0:
        soln = DARPSolution(model.ObjVal)
        model.update_var_values()
        chains, cycles = build_chains(model)
        paths = {}
        assert len(cycles) == 0
        for k, r in enumerate(chains):
            r_frags, r_arcs, _ = r
            paths[k] = tuple(i for f in r_frags for i in f.path)
            print("Vehicle", k)
            if len(r) == 0:
                continue
            print(*map(str, r_frags))
            pprint_path(paths[k], data, add_depots=True)
            assert get_early_schedule(paths[k], data) is not None, "illegal solution"
            soln.add_route(paths[k])
        soln.to_json_file(exp.outputs['solution'])

    print()
    output = TablePrinter(('Section', ' Time (s)'), min_col_width=24)
    times = stopwatch.times.copy()
    times['total'] = sum(times.values())
    for k, v in times.items():
        output.print_line(k, v)

    write_info_file(exp, times,network, obj_info,cut_counter, model)