import time
import matplotlib.pyplot as plt
import sys
import imageio
import glob, os, re


class Timer:
    def __init__(self):
        self.startTime = dict()
        self.timerSummary = dict()

    def startTimer(self, functionName):
        self.startTime[functionName] = time.time()

    def stopTimer(self, functionName):
        if functionName in self.timerSummary:

            self.timerSummary[functionName] += time.time() - self.startTime[functionName]
        else:
            self.timerSummary[functionName] = time.time() - self.startTime[functionName]

    def printTimerSummary(self):
        print("_________________________________________________________")
        print("                        ")
        print("Timer")
        print("_________________________________________________________")
        for x in self.timerSummary:
            print(x, ": ", self.timerSummary[x])
        print("_________________________________________________________")

class Plotter:
    def __init__(self, path, outputName, step, option="skip", fps=20):

        self.path = path
        self.outputName = outputName
        self.step = step
        self.option = option
        self.fps = fps

    def plot(self,it, x, y, con):

        if(it%self.step==0):

            if(self.option == "gif"):
                fig, ax = plt.subplots()
                plt.triplot(x, y, con)
                plt.gca().set_aspect('equal', adjustable='box')

                name = self.path + "/" + self.outputName + '_' + str(it) + '.png'

                fig.savefig(name)
                plt.close(fig)

            if(self.option == "plot"):

                plt.clf()
                plt.triplot(x, y, con)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.draw()
                plt.pause(0.0001)

    def show(self, x, y, con):
        """ save figure as png or plot figure

            :param x: x-coordinates
            :param y: y-coordinates
            :param con: grid connectivity
        """

        plt.clf()
        plt.triplot(x, y, con)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def sorted_alphanumeric(self, data):
        """ numerical sort of the data files

            :param data: data files to be sorted
        """

        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(data, key=alphanum_key)

    def gif(self):
        """ produce a gif from the png files """

        filenames = self.sorted_alphanumeric(os.listdir(self.path))
        print(filenames)

        with imageio.get_writer(self.path + "/" + self.outputName + ".gif", mode='I', fps=self.fps) as writer:
            for filename in filenames:

                if filename.endswith(".png"):
                    image = imageio.imread(self.path + "/" + filename)
                    writer.append_data(image)

    def removePng(self):
        """ delete all pngs in directory self.path """

        filenames = os.listdir(self.path)

        for filename in filenames:
            if filename.endswith(".png"):
                os.remove(os.path.join(self.path, filename))

