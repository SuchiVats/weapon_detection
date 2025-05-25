
def main(**kwargs):
    from rfdetr.util.model import Model

    print("\n✅ Running training via main() in trainer.py")
    model = Model(**kwargs)
    callbacks = {
        "on_fit_epoch_end": [],
        "on_train_batch_start": [],
        "on_train_end": [],
    }
    model.train(callbacks=callbacks, **kwargs)

def distill(**kwargs):
    print("\n⚠️ Distillation is not implemented yet.")