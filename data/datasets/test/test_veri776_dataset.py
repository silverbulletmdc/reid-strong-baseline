from data.datasets.veri776 import VeRi776


def test_VeRi776():
    dataset = VeRi776(root='../../../datasets')
    assert hasattr(dataset, 'train')
    assert hasattr(dataset, 'query')
    assert hasattr(dataset, 'gallery')
