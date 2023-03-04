from models.training import Runner


def test_training_cpu_one_device():
    Runner.train(
        accelerator='cpu',
        devices=1,
        max_epochs=1,
    )
