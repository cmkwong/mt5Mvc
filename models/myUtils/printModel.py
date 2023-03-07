
def print_dict(data_dict):
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    for key, value in data_dict.items():
        print("{}:\t{:.5f}".format(key, value))

def print_list(data_list):
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    for i, value in enumerate(data_list):
        print("{}:\t{}".format(i, value))

def loss_status(writer, loss, episode, mode='train'):
    """
    :param writer: SummaryWriter from pyTorch
    :param loss: float
    :param episode: int
    :param mode: string "train" / "test"
    """
    writer.add_scalar("{}-episode_loss".format(mode), loss, episode)
    print("{}. {} loss: {:.6f}".format(episode, mode, loss))

def print_at(txt, tg=None, print_allowed=True, reply_markup=None):
    if print_allowed:
        if not tg:
            print(txt)
        else:
            tg.bot.send_message(tg.chat_id, txt, reply_markup=reply_markup)