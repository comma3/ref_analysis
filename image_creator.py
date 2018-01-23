from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import glob


def plotImage(point, im):
    """
    Plots an image at each x and y location.
    """
    x, y = point
    bb = Bbox.from_bounds(x,y,0.6,0.1) # Change figure aspect ratio
    bb2 = TransformedBbox(bb,ax.transData)
    bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)

    bbox_image.set_data(im)
    ax.add_artist(bbox_image)

if __name__ == '__main__':

    markers = {}
    for image in glob.glob('logos/*.png'):
        clean_name = image.split('/')[1].replace('logo', '').replace('.png', '').replace('-', ' ').strip()
        markers[clean_name] = plt.imread(image)

    points = {
                'michigan wolverines' : (6,0.9),
                'minnesota golden gophers': (3,0.2),
                'virginia tech hokies': (4,0.2),
                'indiana hoosiers': (5,0.25),
                'michigan state spartans': (-2,0.14),
                'virginia cavaliers': (-1,0.22),
                'north carolina tar heels': (-3,0.5),
                'clemson tigers': (3,0.1),
                'nc state wolfpack': (0,0.9),
                'wake forest demon deacons': (1,0.6),
                'iowa hawkeyes': (0,0.4),
                'penn state nittany lions': (-1,0.35),
                'georgia tech yellow jackets': (1.3,0.45),
                'maryland terrapins': (1.4,0.67),
                'illinois fighting illini': (-3.7,0.75),
                'northwestern wildcats': (-5.1,0.25),
                'syracuse orange': (2.5,0.12),
                'miami hurricanes': (-3.8,0.25),
                'duke blue devils': (3.2,0.8),
                'pittsburgh panthers': (1.1,0.05),
                'louisville cardinals': (-1.76,0.3),
                'wisconsin badgers': (1.6,0.3),
                'ohio state buckeyes': (6,0.4),
                'nebraska cornhuskers': (1.8,0.7),
                'boston college eagles': (2.5,.55),
                'purdue boilermakers': (-2.5,0.76),
                'rutgers scarlet knights': (1.3,0.1),
                'florida state seminoles': (1.1,0.9),
             }


    # Create figure
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    for k, v in points.items():
        plotImage(v, markers[k])

    # Set the x and y limits
    ax.set_ylim(0,1.1)
    ax.set_xlim(-6,7)

    plt.xlabel('Average Call Differential').set_fontsize(26)
    plt.ylabel('Argumentativeness').set_fontsize(26)

    plt.show()
