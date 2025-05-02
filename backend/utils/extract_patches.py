from utils.patch_extractor_casia import PatchExtractorCASIA
from utils.patch_extractor_nc import PatchExtractorNC


def extract_patches(input_path, output_path, dataset_type='casia', patches_per_image=2, stride=128, rotations=4, mode='rot'):
    """
    Extract patches from images for training or testing
    
    Args:
        input_path: Path to the input dataset
        output_path: Path to save the extracted patches
        dataset_type: Type of dataset ('casia' or 'nc')
        patches_per_image: Number of patches to extract per image
        stride: Stride for sliding window
        rotations: Number of rotations to apply
        mode: Rotation mode ('rot' or 'no_rot')
    """
    if dataset_type.lower() == 'casia':
        pe = PatchExtractorCASIA(
            input_path=input_path, 
            output_path=output_path,
            patches_per_image=patches_per_image, 
            stride=stride, 
            rotations=rotations, 
            mode=mode
        )
    elif dataset_type.lower() == 'nc':
        pe = PatchExtractorNC(
            input_path=input_path, 
            output_path=output_path,
            patches_per_image=patches_per_image, 
            stride=stride, 
            rotations=rotations, 
            mode=mode
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    pe.extract_patches()
    return output_path


# Example usage
if __name__ == "__main__":
    # CASIA Dataset
    # mode='no_rot' for no rotations
    pe = PatchExtractorCASIA(input_path='../data/CASIA2', output_path='patches_casia_with_rot',
                            patches_per_image=2, stride=128, rotations=4, mode='rot')
    pe.extract_patches()

    # NC16 Dataset
    # mode='no_rot' for no rotations
    # pe = PatchExtractorNC(input_path='../data/NC2016/', output_path='patches_nc_with_rot',
    #                      patches_per_image=2, stride=32, rotations=4, mode='rot')
    # pe.extract_patches()
