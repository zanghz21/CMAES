import os
import pickle as pkl

def load_archive_from_reload(logdir: str, is_em=False, is_cma_mae=True):
    with open(os.path.join(logdir, 'reload.pkl'), "rb") as file:
        data = pkl.load(file)

        if is_em:
            archive = data["archive"]
            result_archive = None
        else:
            scheduler = data["scheduler"]
            archive = scheduler.archive
            if is_cma_mae:
                result_archive = scheduler.result_archive
            else:
                result_archive = None
        return archive, result_archive
    
logdir='logs/2024-05-27_19-43-47_trafficmapf-sortation-small-linear_CtCaYD5W'
archive, result_archive = load_archive_from_reload(logdir)
df = archive.as_pandas(include_solutions=True, include_metadata=True)

os.makedirs(os.path.join(logdir, 'archive'), exist_ok=True)
df.to_pickle(os.path.join(logdir, f"archive/archive_.pkl"))