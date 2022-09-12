choices = ["Add a new image", "Train VGGNet", "Recognize using VGGNet", "Exit"]
choice = 0
base_folder = 'database/'
dsize = (128, 128)
rows = dsize[0]
cols = dsize[1]
folder_names = []
for entry_name in os.listdir(base_folder):
    entry_path = os.path.join(base_folder, entry_name)
    if os.path.isdir(entry_path):
        folder_names.append(entry_name)