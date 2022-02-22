import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pytorch_model_summary import summary

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen(env):
    # convert 800 * 1200 * 3 to 400 * 600 * 3
    screen = env.render(mode = 'rgb_array').transpose((2,0,1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width) # slice 처리가 가능하도록 index를 정의
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # cart를 중심으로 정사각형 이미지로 변경
    screen = screen[:,:, slice_range]

    # continous한 memory 형태로 반환
    screen = np.ascontiguousarray(screen, dtype = np.float32)
    screen = torch.from_numpy(screen)

    return resize(screen).unsqueeze(0)

def plot_durations(episode_duration):
    plt.figure(1)
    plt.clf()

    durations_t = torch.tensor(episode_duration, dtype = torch.float)

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        
def plot_model_struture(model, input_shape):
    x = torch.zeros(input_shape)
    # 앞에서 생성한 model에 Input을 x로 입력한 뒤 (model(x))  graph.png 로 이미지를 출력합니다.
    # make_dot(model(x), params=dict(model.named_parameters())).render("graph", format="png")
    print(summary(model, x, show_input = True))