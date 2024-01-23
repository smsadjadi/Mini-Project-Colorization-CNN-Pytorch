

from torchvision import transforms
from skimage import color
from PIL import Image
import glob

def MakeDataset(Directory,UsedDataSize='ALL'):

    def ImageOpen(image_path,new_size=(256,256)):
        image = Image.open(image_path).resize(new_size)
        if len(image.split()) != 3: image = image.convert('RGB')
        image = color.rgb2lab(image)
        image_L = transforms.ToTensor()(image[:,:,0])
        image_ab = transforms.ToTensor()(image[:,:,1:])
        return image_L, image_ab

    class MyDataset:
      def __init__(self, imgs_list):
          super(MyDataset, self).__init__()
          self.imgs_list = imgs_list
          self.transforms = transforms
      def __getitem__(self, index):
          image_path = self.imgs_list[index]
          image, target = ImageOpen(image_path)
          return image, target
      def __len__(self):
          return len(self.imgs_list)

    DIR = Directory
    imgs = glob.glob(DIR + '/*.jpg') ; TotalDataSize = len(imgs)
    print("Total images:", TotalDataSize)

    if UsedDataSize == 'ALL': UsedDataSize = TotalDataSize
    dataset = MyDataset(imgs_list = imgs[:UsedDataSize+1])
    print("Used images:", UsedDataSize)
    
    return dataset

