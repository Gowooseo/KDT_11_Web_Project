import cgi, sys, codecs, cgitb
import datetime
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

# 모델과 클래스 매핑 딕셔너리를 로드합니다.
class_mapping = class_mapping ={0:'팔대산인',
 1:'저수량',
 2:'범문강',
 3:'관준',
 4:'황정견',
 5:'홍일',
 6:'유공권',
 7:'양추생',
 8:'루쉰',
 9:'미불',
 10:'마오쩌둥',
 11:'구양순',
 12:'손과정',
 13:'송휘종',
 14:'사맹해',
 15:'왕희지',
 16:'문징명',
 17:'우우임',
 18:'안진경',
 19:'조맹부'}  

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        path,label=self.path_imgs[idx]
        img=Image.open(path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image,label

class callamodel(nn.Module):
    def __init__(self):
        super(callamodel,self).__init__()
        #모델 아키텍처 정의
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 20)
        self.fc3 = nn.Linear(20, 15)
        self.fc4 = nn.Linear(15, len(class_mapping))
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

model = torch.load('./cgi-bin/model.pth', map_location='cpu')

# 이미지 전처리 함수 정의
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # 여기에서 data_transforms를 정의해야 합니다.

    # 데이터 전처리
    data_transforms=transforms.Compose([transforms.RandomRotation(10),transforms.RandomHorizontalFlip(),transforms.CenterCrop(224),transforms.Resize(224),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])])
    
    image_tensor = data_transforms(image).unsqueeze(0)
    return image_tensor

# 이미지 분류 함수 정의
def classify_image(image_path):
    image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()
        predicted_class = class_mapping[predicted_idx]  # 매핑 딕셔너리 사용

    return predicted_class

# 웹페이지의 form 태그 내의 input 태그 입력값 가져와서 저장하고 있는 인스턴스
form = cgi.FieldStorage()

# 클라이언트의 요청 데이터 추출
if 'image' in form:
    fileitem = form['image'] 
    # 서버에 이미지 파일 저장
    img_file = fileitem.filename 
    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    save_path = f'./img/{suffix}_{img_file}'
    with open(file=save_path, mode='wb') as f:
        f.write(fileitem.file.read())

    img_path = f"./img/{suffix}_{img_file}"

    # 이미지를 분류합니다.
    predicted_class = classify_image(save_path)
else:
    img_path = "None"
    predicted_class = "None"

# Web 인코딩 설정 한글 안깨지게 하는 코드(sys.stdout)
sys.stdout = codecs.getwriter(encoding='utf-8')(sys.stdout.detach())


# 요청에 대한 응답 HTML
with open('./html/calla.html', 'r', encoding='utf-8') as f:
    print("Content-Type:text/html ; charset=utf-8")
    print()
    print(f.read())
    print("<Title> CGI script output</TITLE>")
    print("<H1>잘생긴 글자 만들기</H1>")
    print(f"<img src='{'.'+img_path}' width='100' height='100'>")  # 사용자가 넣은 이미지 파일 출력
    print(f"<p>당신과 가장 닮은 글씨체는 {predicted_class}체 입니다</p>")  # 예측된 클래스 출력