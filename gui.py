from tkinter import *

class GUI:
    def __init__(self, game, delay=200):
        """ Initializes a 2-D grid for the game. """
        self.delay = delay
        self.game = game
        self.root = Tk()
        self.root.configure(background="black")
        self.cells = [[None for c in range(self.game.dim[1])]
                for r in range(self.game.dim[0])]
        self.initGUI()
        self.root.after(self.delay + 10000, self.step)
        self.root.mainloop()

    def initGUI(self):
        for r in range(self.game.dim[0]):
            for c in range(self.game.dim[1]):
                self.cells[r][c] = Canvas(self.root, 
                        bg=self.game.grid[r][c].getColor(), width=7,
                        height=7, bd=0)
                self.cells[r][c].configure(highlightthickness=0)
                self.cells[r][c].grid(row=r, column=c)
                
    def step(self):
        self.game.evolve2D()
        for r in range(self.game.dim[0]):
            for c in range(self.game.dim[1]):
                self.cells[r][c]["bg"] = self.game.grid[r][c].getColor()
        self.root.after(self.delay, self.step)