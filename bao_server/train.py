import storage
from storage import CFG_FILE_PATH, read_progress
import model
import model_gnto
from config import read_config
import os
import shutil
import reg_blocker

def get_regressor_class():
    config = read_config()
    model_type = config.get("ModelType", "BAO")
    if model_type == "GNTO":
        return model_gnto.GntoRegression
    return model.BaoRegression

class BaoTrainingException(Exception):
    pass

def train_and_swap(fn, old, tmp, verbose=False,iteration=None):
    Regressor = get_regressor_class()
    if not iteration:
        if os.path.exists(fn):
            old_model = Regressor(have_cache_data=True)
            old_model.load(fn)
        else:
            old_model = None

        new_model = train_and_save_model(tmp, verbose=verbose)
        max_retries = 5
        current_retry = 1
        while not reg_blocker.should_replace_model(old_model, new_model):
            if current_retry >= max_retries == 0:
                print("Could not train model with better regression profile.")
                return
            
            print("New model rejected when compared with old model. "
                + "Trying to retrain with emphasis on regressions.")
            print("Retry #", current_retry)
            new_model = train_and_save_model(tmp, verbose=verbose,
                                            emphasize_experiments=current_retry)
            current_retry += 1

        if os.path.exists(fn):
            shutil.rmtree(old, ignore_errors=True)
            os.rename(fn, old)
        os.rename(tmp, fn)
    else:
        assert type(iteration) == int and iteration >= 0
        if os.path.exists(fn):
            old_model = Regressor(have_cache_data=True)
            old_model.load(fn)
        else:
            old_model = None

        new_model = train_and_save_model_iteration(tmp, iteration, verbose=verbose)
        max_retries = 5
        current_retry = 1
        while not reg_blocker.should_replace_model(old_model, new_model):
            if current_retry >= max_retries == 0:
                print("Could not train model with better regression profile.")
                return
            
            print("New model rejected when compared with old model. "
                + "Trying to retrain with emphasis on regressions.")
            print("Retry #", current_retry)
            new_model = train_and_save_model_iteration(tmp, iteration, verbose=verbose,
                                            emphasize_experiments=current_retry)
            current_retry += 1

        if os.path.exists(fn):
            shutil.rmtree(old, ignore_errors=True)
            os.rename(fn, old)
        os.rename(tmp, fn)

# Qihan: add train_no_swap function
def train_no_swap(fn, verbose=False):
    """
    训练当前模型并直接更新模型参数，但不进行模型替换过程。
    
    参数：
      fn: 模型保存的目标路径（可能为文件或目录）
      verbose: 是否启用详细输出
    
    过程：
      1. 使用 train_and_save_model_episode() 在临时路径中训练新模型。
      2. 删除原有模型文件或目录（如果存在）。
      3. 将临时模型重命名为目标路径 fn，从而完成模型更新。
    """
    tmp = fn + ".tmp"
    
    # Pre-copy the model to tmp so train_and_save_model_episode can load and fine-tune it
    if os.path.exists(tmp):
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        else:
            os.remove(tmp)

    if os.path.exists(fn):
        try:
            if os.path.isdir(fn):
                shutil.copytree(fn, tmp)
            else:
                shutil.copy(fn, tmp)
            if verbose:
                print(f"Copied existing model to {tmp} for fine-tuning.")
        except Exception as e:
            print(f"Failed to copy existing model: {e}. Will train from scratch.")

    new_model = train_and_save_model_episode(tmp, verbose=verbose)
    if os.path.exists(fn):
        if os.path.isdir(fn):
            shutil.rmtree(fn)
        else:
            os.remove(fn)
    os.rename(tmp, fn)
    print("Model updated without swap after each episode.")
    return new_model

def train_and_save_model(fn, verbose=True, emphasize_experiments=0):
    all_experience = storage.experience()

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    Regressor = get_regressor_class()
    reg = Regressor(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    return reg

def train_and_save_model_iteration(fn, iteration, verbose=True, emphasize_experiments=0):
    all_experience = storage.experience_iteration(iteration)

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    Regressor = get_regressor_class()
    reg = Regressor(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    return reg


def train_and_save_model_episode(fn, verbose=True):
    iteration, episode = read_progress()
    all_experience = storage.experience_episode(iteration, episode)
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    if not all_experience:
        raise BaoTrainingException("Cannot episode train a Bao model with no experience")
    
    Regressor = get_regressor_class()
    reg = Regressor(have_cache_data=True, verbose=verbose)

    if os.path.exists(fn):
        try:
            reg.load(fn)
            if verbose:
                print(f"Loaded model from {fn} for fine-tuning.")
        except Exception as e:
            print(f"Warning: Could not load model from {fn}: {e}. Starting from scratch.")

    # Qihan: just light train
    reg.fit(x, y, epochs=20)
    reg.save(fn)
    return reg

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1])

    print("Model saved, attempting load...")
    Regressor = get_regressor_class()
    reg = Regressor(have_cache_data=True)
    reg.load(sys.argv[1])

