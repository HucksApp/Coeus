# Import your class
from Models.identification import CoeusIdentification
from data_sanitation.CheckImage import removeCorruptedImages

if __name__ == '__main__':
    training = True
    data_dir = './data/car_parts'
    save_dir = './Models/trained/identify'
    test_image = './test_data/test33.jpeg'

    if training:
        #removeCorruptedImages(data_dir)
        model = CoeusIdentification(training=True, dataset_path=data_dir, save_dir=save_dir)
        model.to(model.device)
        model.train_in_progressive(epochs_per_run=2)
    else:
        model = CoeusIdentification(training=False, dataset_path=data_dir, save_dir=save_dir)
        model.predict_image(test_image)