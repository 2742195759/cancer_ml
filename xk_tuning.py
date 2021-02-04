import multiprocessing as mp
import numpy as np
import copy
import os
import time

class GpuSchedular:
    def __init__(self, n_gpus=2):
        self.n_gpus = 2
        self.reset()
        
    def allocate(self, task_idx):
        assert(task_idx not in self.task2gpu)
        n_tasks = [len(val) for key, val in self.gpu2tasks.items()]
        idx = (np.array(n_tasks)*-1).argmax()  
        target = [i for i in self.gpu2tasks.keys()][idx] 
        self.gpu2tasks[target].append(task_idx)
        self.task2gpu[task_idx] = target
        return target
    
    def reset(self):
        self.gpu2tasks = {i: [] for i in range(self.n_gpus)}
        self.task2gpu = {}

class TaskPool:
    def __init__(self, n_pool=5, arg_type='long', n_gpu=2, min_val=0.8, max_val=0.9):
        self.n_pool = n_pool
        self.arg_type = arg_type
        self.gpu_schedular = GpuSchedular(n_gpu)
        self.min_val = min_val
        self.max_val = max_val
        self.reset()
    
    def reset(self):
        self.best_param_dict = {}
    
    def collect_results(self, info):
        """
        key, val ? 
        """
        return 0, 12
    
    def _worker(self, param_dict, gpu_id):
        cmd = "CUDA_VISIBLE_DEVICES={}".format(gpu_id) + " " + self._cmd_body() + " " + self._paramdict2str(param_dict)
        print (cmd + "\n")
        r = os.popen(cmd)
        info = r.readlines()
        return self.collect_results(info)
        
    def start(self, grid_param_dict):
        """
        grad_param_dict: 
            'anchor_model: [4]'
            'reg': [0.001, 0.01, 0.1]
        """
        s = time.time()
        for key, vals in grid_param_dict.items():
            results = [None for _ in range(len(vals))]
            with mp.Pool(self.n_pool) as p:
                for idx, val in enumerate(vals):
                    tmp_dict = copy.deepcopy(self.best_param_dict)
                    tmp_dict.update({key:val})
                    results[idx] = p.apply_async(self._worker, (tmp_dict, self.gpu_schedular.allocate(idx)))
                
                output_keys = [None for _ in range(len(vals))]
                output_vals = [None for _ in range(len(vals))]
                for idx, res in enumerate(results): 
                    output_keys[idx], output_vals[idx] = res.get()
                self.gpu_schedular.reset()
                
                for ret_idx, (ret_key, ret_val) in enumerate(zip(output_keys, output_vals)):
                    print ("[Summary]{}:{}:\n\t{}".format(key, vals[ret_idx], ret_val))

                valid_ids = []
                for ret_idx, key_val in enumerate(output_keys):
                    if self.min_val < key_val < self.max_val: 
                        valid_ids.append(ret_idx)

                print(valid_ids)
                for idx in valid_ids:
                    print ("Find Better {}={}:\n {}".format(key, vals[idx], output_vals[idx]))
                self.best_param_dict[key] = vals[valid_ids[0]]
                
        print ('Time: \t', (time.time() - s), ' sec')
        print ("Best Parameters: \n{}".format(self.best_param_dict))

    def _paramdict2str(self, dic):
        params = []
        slash = "--" if self.arg_type == "long" else "-"
        for key, val in dic.items():
            assert (type(key) == str)
            params.append(slash + key)
            params.append(str(val))
        return " ".join(params)
    
    def _cmd_body(self):
        return "head main.py"
        
    
class CancerTaskPool(TaskPool):
    def __init__(self, n_pool, arg_type, min_val=0.8, max_val=0.9):
        super(CancerTaskPool, self).__init__(n_pool, arg_type, 2, min_val, max_val)
        
    def _cmd_body(self):
        #return "python main.py"
        return "python3 preprocess.py "
    
    def collect_results(self, info):
        parameter, result, result_reward, time_cost = '', '', '', ''
        useful_info = []
        result = ''
        for idx, line in enumerate(info):
            if 'AUC' in line:
                result = line
                useful_info = info[idx-7:idx+1]

        if result != "": 
            auc = float(result.split(':')[1])
        return auc, "".join(useful_info)

if __name__ == "__main__":
    from config import interested_key 
    param_list = {
        'random_state': range(1, 100),
        'l1_reg': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    }
    min_val = 0.85
    max_val = 0.94
    taskpool = CancerTaskPool(5, 'long', min_val, max_val)
    taskpool.start(param_list)
