import orbax.checkpoint as ocp, flax.nnx as nnx, yaml
from pathlib import Path
from safe_jepa.nn.variables import NormStat, SpectralStat
from safe_jepa.world_model.world_model import WorldModel

INFERENCE_FILTER = nnx.Any(nnx.Param, NormStat, SpectralStat)

def save(run_dir, world_model, cfg):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    ckpt = ocp.StandardCheckpointer()
    ckpt.save(run_dir / "model", nnx.state(world_model, INFERENCE_FILTER))
    ckpt.save(run_dir / "train", nnx.state(world_model))
    ckpt.wait_until_finished()


def restore(run_dir, obs_mean, obs_std, *, for_training=False):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    model = nnx.eval_shape(lambda: WorldModel(cfg, obs_mean, obs_std, rngs=nnx.Rngs(0)))
    graphdef, abstract = nnx.split(model)
    which = "train" if for_training else "model"
    flt = None if for_training else INFERENCE_FILTER
    restored = ocp.StandardCheckpointer().restore(
        run_dir / which, nnx.state(model, flt) if flt else abstract
    )
    nnx.update(model, restored)
    return model, cfg