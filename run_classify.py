# Import your class
from Models.classification import CoeusClassification
from data_sanitation.CheckImage import removeCorruptedImages

if __name__ == '__main__':
    training = True
    data_dir = './data/Toyota/toyota_cars'
    save_dir = './Models/trained'
    test_image = './test_data/test33.jpeg'

    if training:
        #removeCorruptedImages(data_dir)
        model = CoeusClassification(training=True, dataset_path=data_dir, save_dir=save_dir)
        model.to(model.device)
        model.train_in_progressive(epochs_per_run=2)
    else:
        model = CoeusClassification(training=False, dataset_path=data_dir, save_dir=save_dir)
        model.predict_image(test_image)