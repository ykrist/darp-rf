from oru import *
from oru.slurm import slurm_format_time
from typing import FrozenSet, Tuple, Union
from utils import *
from utils.data import DARP_Data, get_named_instance_DARP, modify, indices, get_name_by_index, SDARP_Data
import yaml
from gurobi import *
import json
frozen_dataclass = dataclasses.dataclass(frozen=True, eq=True)

def print_when_used(args=True, result=True):
    p_args = args
    p_result=result
    def decorator(func):
        def wrapper(*args, **kwargs):
            info = "called " + func.__name__
            if p_args:
                info+=("(" +
                      ",".join(itertools.chain(map(str,args),map(lambda kv : f"{kv[0]!s}={kv[1]!s}", kwargs.items())))
                       + ")")
            else:
                info+= '(...)'
            retval = func(*args, **kwargs)
            if p_result:
                info+=f' - returned {retval!s}'
            print(info)
            return retval
        return wrapper
    return decorator

def locstring(i, num_req):
    if i == 0:
        return ' O+'
    elif 1 <= i <= num_req:
        return f'{i:2d}P'
    elif num_req + 1 <= i <= 2 * num_req:
        return f'{i - num_req:2d}D'
    else:
        return ' O-'


def _format_fragment(f):
    return '--'.join((str(f.start),) + tuple(map(str,f.path[1:-1])) + (str(f.end),))

def _format_arc(a):
    return str(a.start) + '--' + str(a.end)

@frozen_dataclass
class LNode(SerialisableFrozenSlottedDataclass, LazyHashFrozenDataclass):
    __slots__ = ['loc', 'load']
    loc : int
    load : FrozenSet[int]

    def __str__(self):
        return '({:d},{:s})'.format(self.loc,'{' + ','.join(str(i) for i in self.load) + '}')

@frozen_dataclass
class LArc(SerialisableFrozenSlottedDataclass, LazyHashFrozenDataclass):
    __slots__ = ['start', 'end']
    start : LNode
    end : LNode

    __str__ = _format_arc

@frozen_dataclass
class ResFrag(SerialisableFrozenSlottedDataclass, LazyHashFrozenDataclass):
    __slots__ = ['start', 'end', 'path']
    start : LNode
    end : LNode
    path : Tuple[int]

    __str__ = _format_fragment


@frozen_dataclass
class Frag(SerialisableFrozenSlottedDataclass, LazyHashFrozenDataclass):
    __slots__ = ['start', 'end', 'path']
    start : int
    end : int
    path : Tuple[int]

    __str__ = _format_fragment


def copy_delitem(d: dict, key):
    d = d.copy()
    del d[key]
    return d


def copy_additem(d: dict, key, value):
    d = d.copy()
    d[key] = value
    return d


def tighten_time_windows(data : Union[SDARP_Data, DARP_Data]) -> Union[DARP_Data, SDARP_Data]:
    o_depot = 0
    d_depot = 2*data.n + 1
    tw_start = dict()
    tw_end = dict()

    for p in data.P:
        d = p + data.n
        tw_start[p] = max(data.tw_start[p],
                          data.tw_start[d] - data.max_ride_time[p],
                          data.tw_start[o_depot] + data.travel_time[o_depot, p])

        tw_end[d] = min(data.tw_end[d],
                        data.tw_end[p] + data.max_ride_time[p],
                        data.tw_end[d_depot] - data.travel_time[d, d_depot])

        tw_end[p] = min(data.tw_end[p], tw_end[d] - data.travel_time[p, d])
        tw_start[d] = max(data.tw_start[d], tw_start[p] + data.travel_time[p, d])


    for i in (o_depot, d_depot):
        tw_start[i] = data.tw_start[i]
        tw_end[i] = data.tw_end[i]

    return dataclasses.replace(data, tw_start=frozendict(tw_start), tw_end=frozendict(tw_end))



def remove_arcs(data: Union[DARP_Data, SDARP_Data]) -> Union[SDARP_Data,DARP_Data]:
    o_depot = 0
    d_depot = 2*data.n + 1
    N = range(d_depot+1)

    illegal_arcs = {
        *((o_depot,j) for j in data.D),
        *((i,d_depot) for i in data.P),
        *((i+data.n,i) for i in data.P),
        *((i,i) for i in N),
        *((i, o_depot) for i in N),
        *((d_depot, i) for i in N),
    }

    for i,j in itertools.combinations(data.P, 2):
        if data.demand[i] + data.demand[j] > data.capacity + EPS:
            illegal = [True]*4
            paths_to_check = []
        else:
            illegal = []
            paths_to_check = [
                (i, j, j + data.n, i + data.n),
                (i, j, i + data.n, j + data.n),
                (j, i, i + data.n, j + data.n),
                (j, i, j + data.n, i + data.n),
            ]

        paths_to_check.extend([
            (i, i + data.n, j, j + data.n),
            (j, j + data.n, i, i + data.n)
        ])

        illegal.extend([get_early_schedule(p, data) is None for p in paths_to_check])

        if illegal[0] and illegal[1]:
            illegal_arcs.add((i, j))

        if illegal[2] and illegal[3]:
            illegal_arcs.add((j, i))

        if illegal[1] and illegal[2]:
            illegal_arcs.add((i + data.n, j + data.n))

        if illegal[0] and illegal[3]:
            illegal_arcs.add((j + data.n, i + data.n))

        if illegal[1]:
            illegal_arcs.add((j, i + data.n))

        if illegal[3]:
            illegal_arcs.add((i, j + data.n))

        if illegal[4]:
            illegal_arcs.add((i + data.n, j))

        if illegal[5]:
            illegal_arcs.add((j + data.n, i))

    if isinstance(data, SDARP_Data):
        return dataclasses.replace(
            data,
            travel_time=frozendict((a, v) for a, v in data.travel_time.items() if a not in illegal_arcs),
        )
    else:
        assert isinstance(data, DARP_Data)
        return dataclasses.replace(
            data,
            travel_time=frozendict((a, v) for a, v in data.travel_time.items() if a not in illegal_arcs),
            travel_cost=frozendict((a, v) for a, v in data.travel_cost.items() if a not in illegal_arcs),
        )

def _dmap(i:int, n:int):
    if i == 0:
       return 2*n+1
    return i+n

def _pmap(i : int, n : int):
    if i == 2*n+1:
        return 0
    return i-n

def get_early_schedule(path, data: DARP_Data, start_time=0, check_illegal=True):
    #adapted from Hunsaker and Savelsbergh 2002, Tang 2010
    # the difference here is that we don't have a maximum waiting time.

    # need to add on the depot nodes to check max route duration.
    o_depot_loc = 0
    d_depot_loc = 2*data.n+1
    path = (o_depot_loc, ) + path + (d_depot_loc,)
    start_time = max(0, start_time - data.travel_time[0,path[1]])

    arrival = dict()
    srv_start = dict()
    # pass 1 (forward pass)
    pairwise_arcs = list(zip(path, path[1:]))
    path_set = set(path)
    arrival[path[0]] = max(data.tw_start[path[0]], start_time)
    srv_start[path[0]] = arrival[path[0]]
    illegal = False
    for i,j in pairwise_arcs:
        arrival[j] = srv_start[i] + data.travel_time[i,j]
        srv_start[j] = max(arrival[j], data.tw_start[j])

        if arrival[j] > data.tw_end[j] + EPS:
            if check_illegal:
                return None
            else:
                illegal = True

    if illegal:
        return tuple(srv_start[i] for i in path[1:-1])

    # Pass 2 backward pass
    cum_waiting_time = srv_start[path[-1]] - arrival[path[-1]]
    idx = len(path)-1
    for i,j in reversed(pairwise_arcs):
        idx -= 1
        srv_start[i] = arrival[j] - data.travel_time[i,j]
        if i in data.P or i == o_depot_loc:
            i_d = _dmap(i,data.n)
            if i_d in path_set:
                delta = srv_start[i_d] - srv_start[i] - data.max_ride_time[i]
                if delta > 0:
                    if delta > cum_waiting_time + EPS:
                        if check_illegal:
                            return None
                        else:
                            return tuple(srv_start[i] for i in path[1:-1])
                    else:
                        srv_start[i] += delta
                        cum_waiting_time -= delta
                        for k,l in zip(path[idx:], path[idx+1:]):
                            srv_start[l] = max(srv_start[l], srv_start[k]+data.travel_time[k,l])


                if srv_start[i] > data.tw_end[i] + EPS:
                    if check_illegal:
                        return None
                    else:
                        return tuple(srv_start[i] for i in path[1:-1])
        cum_waiting_time += srv_start[i] - arrival[i]

    # Pass 3 (final forward pass)
    for i,j in pairwise_arcs:
        arrival[j] = srv_start[i] + data.travel_time[i,j]
        srv_start[j] = max(srv_start[j], arrival[j])

        if srv_start[j] > data.tw_end[j] + EPS:
            if check_illegal:
                return None
            else:
                return tuple(srv_start[i] for i in path[1:-1])

        if j in data.D or j == d_depot_loc:
            j_p = _pmap(j, data.n)
            if j_p in path_set:
                if srv_start[j] - srv_start[j_p] > data.max_ride_time[j_p] + EPS:
                    if check_illegal:
                        return None
                    else:
                        return tuple(srv_start[i] for i in path[1:-1])

    schedule = tuple(srv_start[i] for i in path[1:-1])
    return schedule


def get_late_schedule(path, data : DARP_Data, end_time=float('inf')):
    # Determines the latest schedule for a path that starts service at the last node at `t=end_time` or earlier.
    departure = dict()
    srv_start = dict()
    # pass 1 (backward pass)
    pairwise_arcs = list(zip(path, path[1:]))
    path_set = set(path)
    departure[path[-1]] = min(data.tw_end[path[-1]], end_time)
    srv_start[path[-1]] = departure[path[-1]]
    # illegal = False
    for i,j in reversed(pairwise_arcs):
        departure[i] = srv_start[j] - data.travel_time[i,j]
        srv_start[i] = min(departure[i], data.tw_end[i])

        if srv_start[i] + EPS < data.tw_start[i]:
            return None

    # pass 2
    cum_waiting_time = departure[path[0]] - srv_start[path[0]]
    idx = 0
    for i,j in pairwise_arcs:
        idx += 1
        srv_start[j] = departure[i] + data.travel_time[i,j]
        if j in data.D:
            jp = _pmap(j,data.n)
            if jp in path_set:
                delta = srv_start[j] - srv_start[jp] - data.max_ride_time[jp]
                if delta > EPS:
                    if delta > cum_waiting_time + EPS:
                        return None
                    else:
                        srv_start[j] -= delta
                        cum_waiting_time -= delta
                        for k in reversed(range(idx)):
                            a,b = path[k:k+2]
                            srv_start[a] = min(srv_start[a], srv_start[b] - data.travel_time[a,b])
                            # print(srv_start[k], srv_start[l] - data.travel_time[k,l])

            if srv_start[i] + EPS < data.tw_start[i]:
                return None
        cum_waiting_time += (departure[j] - srv_start[j])

    # pass 3
    for i,j in reversed(pairwise_arcs):
        departure[i] = srv_start[j] - data.travel_time[i,j]
        srv_start[i] = min(departure[i], srv_start[i])

        if srv_start[i] + EPS < data.tw_start[i]:
            return None

        if i in data.P:
            i_d = _dmap(i,data.n)
            if i_d in path_set:
                if srv_start[i_d] - srv_start[i] > data.max_ride_time[i] + EPS:
                    return None

    return tuple(srv_start[i] for i in path)



def schedule_total_wait_time(schedule, path, data : DARP_Data):
    tt = sum(data.travel_time[i,j] for i,j in zip(path, path[1:]))
    return schedule[-1] - schedule[0] - tt


def schedule_ride_times(schedule, path, data):
    onboard = {}
    ride_times = {}
    for i,t in zip(path, schedule):
        if i in data.P:
            onboard[i+data.n] = t
        elif i in onboard:
            ride_times[i-data.n] = t - onboard.pop(i)

    return ride_times


def is_schedule_legal(schedule, path, data : DARP_Data):
    delivery_deadlines = {}
    for i,t in zip(path, schedule):
        if t > data.tw_end[i] + EPS or t + EPS < data.tw_start[i]:
            return False
        elif i in delivery_deadlines and delivery_deadlines[i] > t:
            return False
        elif i in data.P:
            delivery_deadlines[i] = t + data.max_ride_time[i]

    for (i,j), (ti, tj) in zip(zip(path, path[1:]), zip(schedule, schedule[1:])):
        if tj - ti + EPS < data.travel_time[i,j]:
            return False

    return True

def delay_schedule(delay, schedule, path, data : DARP_Data):
    """Apply a delay of `delay` to the start of `schedule`."""
    new_schedule = list(schedule)
    for idx, (i,j,ti,tj) in enumerate(zip(path, path[1:], schedule, schedule[1:])):
        new_schedule[idx] += delay
        wait = max(tj - (ti + data.travel_time[i,j]), 0)
        delay -= wait
        if delay < EPS:
            break
    else:
        new_schedule[-1] += delay

    assert is_schedule_legal(new_schedule, path, data)
    return tuple(new_schedule)
#
# def get_schedule_lp(path, data : DARP_Data, start_time=0):
#     m = Model()
#     m.setParam('OutputFlag', 0)
#     T = { i: m.addVar(lb=data.tw_start[i], ub=data.tw_end[i]) for i in path}
#     m.update()
#     T[path[0]].lb  = max(start_time, T[path[0]].lb)
#     for i in path:
#         if i in data.P and i+data.n in path:
#             m.addConstr(T[i+data.n] - T[i] <= data.service_time[i] + data.max_ride_time)
#     for i,j in zip(path,path[1:]):
#         m.addConstr(T[j]-T[i] >= data.travel_time[i,j])
#
#     m.setObjective(T[path[-1]])
#     m.optimize()
#     if m.status == GRB.INFEASIBLE:
#         return None
#     m.addConstr(m.getObjective() == m.ObjVal)
#     m.setObjective(T[path[-1]]-T[path[0]])
#     m.optimize()
#     return tuple(T[i].X for i in path)

def pformat_path(path, data : DARP_Data, color=True, add_depots=False):
    schedule = get_early_schedule(path, data, check_illegal=False)
    if add_depots:
        O = 0
        D = data.n*2+1
        schedule = (schedule[0] - data.travel_time[O,path[0]],) + schedule + \
                   (schedule[-1] + data.travel_time[path[-1],D], )
        path = (O, ) + path + (D, )

    return pformat_schedule(schedule, path, data, color=color)

def pformat_schedule(schedule, path, data : DARP_Data, color=True):
    output = ''

    cell_width = 8
    h_sep = ' '

    times = schedule

    path_str = map(lambda i: locstring(i, data.n).center(cell_width), path)
    if color:
        colors = [TTYCOLORS.CYAN if i in data.P else (TTYCOLORS.MAGENTA if i in data.D else None) for i in path]
        path_str = [colortext(s, c) for s, c in zip(path_str, colors)]
    output += h_sep.join(path_str) + '\n'

    tw_start = []
    tw_end = []
    times_str = []
    float_fmt = f"{{:{cell_width:d}.3f}}"
    format_float_val = lambda v: float_fmt.format(v)
    delivery_deadlines = {}
    for i, t in zip(path, times):
        if i in data.P:
            delivery_deadlines[i + data.n] = t + data.max_ride_time[i]
        elif i in delivery_deadlines:
            if t <= delivery_deadlines[i] + EPS:
                del delivery_deadlines[i]

    for i, t in zip(path, times):
        ts = format_float_val(t)
        es = format_float_val(data.tw_start[i])
        ls = format_float_val(data.tw_end[i])
        if color:
            if t < data.tw_start[i] - EPS:
                es = colortext(es, TTYCOLORS.RED)
            if t > data.tw_end[i] + EPS:
                ls = colortext(ls, TTYCOLORS.RED)
            if i + data.n in delivery_deadlines or i in delivery_deadlines:
                ts = colortext(ts, TTYCOLORS.RED)
        tw_start.append(es)
        tw_end.append(ls)
        times_str.append(ts)

    output += h_sep.join(tw_start) + '\n'
    output += h_sep.join(times_str) + '\n'
    output += h_sep.join(tw_end)

    return output


def pprint_path(path, data: DARP_Data, add_depots=False):
    print(pformat_path(path, data, add_depots=add_depots))

def pprint_schedule(schedule, path, data : DARP_Data):
    print(pformat_schedule(schedule, path, data))

class DomSet:
    def __init__(self, key_func, partial_ordering):
        self.get_key = key_func
        self.criterion = partial_ordering
        self._groups = defaultdict(set)
        self.num_total = 0

    def add(self, item):
        key = self.get_key(item)
        remove = set()
        self.num_total += 1
        new_item_dominates = False
        for other in self._groups[key]:
            if self.criterion(item, other):
                remove.add(item)
                new_item_dominates = True
            elif not new_item_dominates and self.criterion(other, item):
                return False

        self._groups[key] -= remove
        self._groups[key].add(item)
        return True

    def __iter__(self):
        return itertools.chain(*self._groups.values())

    def itergroups(self):
        return self._groups.items()

    def __len__(self):
        return sum(map(sum, self._groups.values()))

def simple_legality_check(path, data : DARP_Data):
    schedule = [data.tw_start[path[0]]]
    tt = []
    load = data.demand[path[0]]
    for i,j in zip(path, path[1:]):
        load += data.demand[j]
        if load > data.capacity:
            print(f"path {path!s} violates capacity")
            return False
        schedule.append(max(schedule[-1] + data.travel_time[i,j], data.tw_start[j]))
        if schedule[-1] > data.tw_end[j]:
            # print(f"path {path!s} violates time windows")
            return False
        tt.append(data.travel_time[i,j])

    for k1,i in enumerate(path):
        if i in data.P:
            try:
                k2 = path.index(i+data.n)
            except IndexError:
                continue
            else:
                if sum(tt[k1:k2]) > data.max_ride_time[i]:
                    print(f"path {path!s} violates max ride time")
                    return False


    return True

class DARPSolution:
    def __init__(self, final_objective):
        self.obj = final_objective
        self.routes = []

    def add_route(self, path):
        self.routes.append(path)

    def to_json_file(self, filename):
        with open(filename, 'w') as fp:
            json.dump({
                'obj' : self.obj,
                'routes' : sorted(self.routes),
            }, fp, indent='\t')


    @classmethod
    def from_json_file(cls, filename):
        with open(filename, 'r') as fp:
            d = json.load(fp)
        soln = DARPSolution(d['obj'])
        for r in d['routes']:
            soln.add_route(tuple(r))
        return soln

    def __eq__(self, other):
        if not isinstance(other, DARPSolution):
            raise NotImplementedError

        return math.isclose(self.obj, other.obj, rel_tol=1e-6) and self.routes == other.routes

    def pprint(self, data : DARP_Data):
        print(self.pformat(data))

    def pformat(self, data):
        s = ''
        for k,r in enumerate(self.routes):
            s += f"Vehicle {k:d}\n"
            s += pformat_path(r, data) + '\n'
        return s[:-1]


def _read_yaml_file(filename):
    if isinstance(filename, dict):
        return filename
    with open(filename, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CFullLoader)

class DARP_Experiment(BaseExperiment):
    ROOT_PATH = BaseExperiment.ROOT_PATH / 'darp'
    INPUTS = {
        "index": {
            "type": "integer",
            "coerce": int,
            "min": 0,
            "max": 67,
            "help" : "Instances 0-47 are the A- and B- instances.  Instances 48-67 are the harder PR- instances."
        },
        "instance": {
            "type": "string",
            "derived": True,
        },
        "extend" : {
            "type": "integer",
            "min": 0,
            "max": 10,
            "default": 0,
            "help" : "Extend the instance time windows and vehicle capacity by a factor of (1 + EXTEND/3).  "
                     "See Gschwind and Irnich, 2015"
        },
        "ridetime" : {
            "type" : "float",
            "min" : 0,
            "default" : 1,
            "help" : "Modify the ride times of the instances by a factor of `ridetime`."
       },
        **BaseExperiment.INPUTS
    }
    PARAMETERS = {
        "param_name": {
            "type": "string",
            "default": "",
            "help": "Parameter set name, will be generated automatically from a hash of parameters if left blank"
        },
        "timelimit": {
            "type": "float",
            "min": 0,
            "coerce": float,
            "default": 3600
        },
        "gurobi": {
            "type": "dict",
            "default": {},
            "keysrules": {
                "type": "string",
            },
            "valuesrules": {
                "type": ["number", "string"]
            },
            'coerce' : _read_yaml_file,
            'help' : 'Path to JSON/YAML file containing Gurobi parameters'
        },
        "cpus": {
            "type": "integer",
            "min": 1,
            "default": 4
        },
        "debug": {
            "type": "boolean",
            "default": False
        },
        **BaseExperiment.PARAMETERS
    }
    OUTPUTS = {
        "info": {"type": "string", "derived": True, 'coerce' : str},
        "solution": {"type": "string", "derived": True, 'coerce' : str},
        **BaseExperiment.OUTPUTS
    }

    @property
    def parameter_string(self):
        if len(self.parameters['param_name']) > 0:
            return self.parameters['param_name']
        if self.parameters['debug']:
            return 'debug'
        return super().parameter_string

    def __init__(self, inputs, outputs, parameters=None):
        super().__init__(inputs, outputs, parameters)
        self._data = None

    def define_derived(self):
        self.inputs["instance"] = get_name_by_index("darp", self.inputs["index"])
        self.outputs["info"] = self.get_output_path("info.json")
        self.outputs["solution"] = self.get_output_path("soln.json")
        super().define_derived()

    def write_index_file(self):
        index = {k: os.path.basename(v) for k, v in self.outputs.items() if k != 'indexfile'}
        index.update(self.inputs)
        index['data_id'] = self.data.id
        with open(self.outputs['indexfile'], 'w') as fp:
            json.dump(index, fp, indent='\t')
        return index

    @classmethod
    def get_parser_arguments(cls):
        clargs = super().get_parser_arguments()
        args, kwargs = clargs['gurobi']
        kwargs["metavar"] = "FILE"
        clargs['gurobi'] = (args, kwargs)
        return clargs

    @property
    def data(self) -> DARP_Data:
        if self._data is None:
            self._data = get_named_instance_DARP(self.inputs['instance'])
            ex = self.inputs['extend']
            if ex > 0:
                self._data = modify.gschwind_extend(self._data, ex)
            rt = self.inputs['ridetime']
            if rt != 1:
                self._data = modify.ride_times(self._data, rt)
        return self._data


    @property
    def input_string(self):
        if self.inputs['extend'] > 0:
            extend_str = '-EX{extend:d}'.format_map(self.inputs)
        else:
            extend_str = ''
        if self.inputs['ridetime'] != 1:
            ridetime_str = '-R{ridetime:.04f}'.format_map(self.inputs).replace('.', '_')
        else:
            ridetime_str = ''

        s = "{index:d}-{instance}".format_map(self.inputs) + extend_str + ridetime_str
        return s

    @property
    def resource_mail_user(self) -> str:
        return 'yanni555rist@gmail.com'

    @property
    def resource_name(self) -> str:
        return self.input_string

    @property
    def resource_constraints(self) -> str:
        return 'R640'

    @property
    def resource_cpus(self) -> str:
        return str(self.parameters['cpus'])

    @property
    def resource_time(self) -> str:
        return slurm_format_time(self.parameters['timelimit'] + 300)

NAMED_DATASET_INDICES = {
    'all': indices.set_from_ranges_inc([(0, 67)]),
    'easy': indices.set_from_ranges_inc([(0,22), (24,34)]) | frozenset({36}),
    'medium': indices.set_from_ranges_inc([(0,43)]),
    'a' : indices.set_from_ranges_inc([(0,23)]),
    'b' : indices.set_from_ranges_inc([(24,47)]),
    'r' : indices.set_from_ranges_inc([(48,67)]),
    'threads' : indices.set_from_ranges_inc([(43,47)])
}
NAMED_DATASET_INDICES['hard'] = (NAMED_DATASET_INDICES['a'] | NAMED_DATASET_INDICES['b']) - frozenset([47])
