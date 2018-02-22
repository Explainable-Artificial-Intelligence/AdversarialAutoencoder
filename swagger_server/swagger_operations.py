import datetime
import logging

from DataLoading import read_mnist_data_from_ubyte

from connexion import NoContent

# our memory-only pet storage
PETS = {}
data = None


def get_pets(limit, animal_type=None):
    return [pet for pet in PETS.values() if not animal_type or pet['animal_type'] == animal_type][:limit]


def get_pets_test(limit, animal_type=None):
    return [pet for pet in PETS.values() if not animal_type or pet['animal_type'] == animal_type][:limit]


"""
Load functions
"""


def load_data(path_to_data_dir):
    # TODO: other params
    global data
    data = read_mnist_data_from_ubyte(path_to_data_dir)
    return True


def load_single_image(image_id, train_test_or_validation):

    def return_single_image(x):
        return {
            'train': data.train.images[image_id],
            'test': data.test.images[image_id],
            'validation': data.validation.images[image_id]
        }[x]

    # TODO: actually return the image

    print(return_single_image(train_test_or_validation))
    print(type(return_single_image(train_test_or_validation)))

    return True


def load_single_label(label_id, train_test_or_validation):

    def return_single_image(x):
        return {
            'train': data.train.labels[label_id],
            'test': data.test.labels[label_id],
            'validation': data.validation.labels[label_id]
        }[x]

    # TODO: actually return the image

    print(return_single_image(train_test_or_validation))
    print(type(return_single_image(train_test_or_validation)))

    return True


def load():
    return "load"


"""
Build functions
"""


def build():
    return "build"


"""
Train functions
"""


def train():
    return "train"


"""
Visualize functions
"""


def visualize():
    return "visualize"


"""
Tune functions
"""


def tune():
    return "tune"


def get_pet(pet_id):
    pet = PETS.get(pet_id)
    return pet or ('Not found', 404)


def put_pet(pet_id, pet):
    exists = pet_id in PETS
    pet['id'] = pet_id
    if exists:
        logging.info('Updating pet %s..', pet_id)
        PETS[pet_id].update(pet)
    else:
        logging.info('Creating pet %s..', pet_id)
        pet['created'] = datetime.datetime.utcnow()
        PETS[pet_id] = pet
    return NoContent, (200 if exists else 201)


def delete_pet(pet_id):
    if pet_id in PETS:
        logging.info('Deleting pet %s..', pet_id)
        del PETS[pet_id]
        return NoContent, 204
    else:
        return NoContent, 404