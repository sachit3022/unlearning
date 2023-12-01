import torch

# Helper functions for loading the hidden dataset.
def load_example(image,image_id,age_group,age,person_id):
    #age and age group are the same for now and they are indicative of the label class.
    
    result = {
        'image': image.to(torch.float32),
        'image_id': image_id,
        'age_group': age_group,
        'age': age,
        'person_id':person_id
    }
    return (image.to(torch.float32),age_group)