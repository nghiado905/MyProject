package algorithm;

import main.GamePanel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Stack;

public class AAsterisk {
    public GamePanel gp;

    public static class Pair {
        private int first;
        private int second;

        public Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }

        public int getFirst() {
            return first;
        }

        public int getSecond() {
            return second;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof Pair && this.first == ((Pair) obj).first && this.second == ((Pair) obj).second;
        }

        @Override
        public String toString() {
            return "(" + first + "," + second + ")";
        }
    }

    public static class Details {
        double value;
        int i;
        int j;

        public Details(double value, int i, int j) {
            this.value = value;
            this.i = i;
            this.j = j;
        }
    }

    // a Cell (node) structure
    public static class Cell {
        public Pair parent;
        // f = g + h, where h is heuristic
        public double f, g, h;

        public Cell() {
            parent = new Pair(-1, -1);
            f = -1;
            g = -1;
            h = -1;
        }

        public Cell(Pair parent, double f, double g, double h) {
            this.parent = parent;
            this.f = f;
            this.g = g;
            this.h = h;
        }
    }

    private int pathId = 5;

    public AAsterisk(int pathId) {
        this.pathId = pathId;
    }

    // method to check if our cell (row, col) is valid
    boolean isValid(int[][] grid, int rows, int cols, Pair point) {
        if (rows > 0 && cols > 0)
            return (point.first >= 0) && (point.first < rows) && (point.second >= 0) && (point.second < cols);
        return false;
    }


    boolean isUnBlocked(int[][] grid, int rows, int cols, Pair point) {
        return isValid(grid, rows, cols, point) && grid[point.first][point.second] == pathId;
    }

    //Method to check if destination cell has been already reached
    boolean isDestination(Pair position, Pair dest) {
        return position == dest || position.equals(dest);
    }

    // Method to calculate heuristic function
    double calculateHValue(Pair src, Pair dest) {
        return Math.sqrt(Math.pow((src.first - dest.first), 2.0) + Math.pow((src.second - dest.second), 2.0));
    }


    // Method for tracking the path from source to destination

    ArrayList<Pair> tracePath(Cell[][] cellDetails, int cols, int rows, Pair dest) {
        ArrayList<Pair> ans = new ArrayList<>();
        Stack<Pair> path = new Stack<>();

        int row = dest.first;
        int col = dest.second;

        Pair nextNode;
        do {
            path.push(new Pair(row, col));
            nextNode = cellDetails[row][col].parent;
            row = nextNode.first;
            col = nextNode.second;
        } while (cellDetails[row][col].parent != nextNode);


        while (!path.empty()) {
            Pair p = path.peek();
            path.pop();
            ans.add(p);
        }
        return ans;
    }

// A main method, A* Search algorithm to find the shortest path

    public ArrayList<Pair> aStarSearch(int[][] grid, int rows, int cols, Pair src, Pair dest) {
        System.out.println("Running");
        if (!isValid(grid, rows, cols, src)) {
            System.out.println("Source is invalid...");
            return null;
        }


        if (!isValid(grid, rows, cols, dest)) {
            System.out.println("Destination is invalid...");
            return null;
        }


        if (!isUnBlocked(grid, rows, cols, src) || !isUnBlocked(grid, rows, cols, dest)) {
            System.out.println("Source or destination is blocked...");
            return null;
        }


        if (isDestination(src, dest)) {
            System.out.println("We're already there...");
            ArrayList<Pair> ans = new ArrayList<>();
            ans.add(new Pair(src.first, src.second));
        }


        boolean[][] closedList = new boolean[rows][cols]; //closed list

        Cell[][] cellDetails = new Cell[rows][cols];

        int i, j;
        // Initialising of the starting cell
        i = src.first;
        j = src.second;
        cellDetails[i][j] = new Cell();
        cellDetails[i][j].f = 0.0;
        cellDetails[i][j].g = 0.0;
        cellDetails[i][j].h = 0.0;
        cellDetails[i][j].parent = new Pair(i, j);


        // Creating an open list


        PriorityQueue<Details> openList = new PriorityQueue<>((o1, o2) -> (int) Math.round(o1.value - o2.value));

        // Put the starting cell on the open list,   set f.startCell = 0

        openList.add(new Details(0.0, i, j));

        while (!openList.isEmpty()) {
            Details p = openList.peek();
            // Add to the closed list
            i = p.i; // second element of tuple
            j = p.j; // third element of tuple

            // Remove from the open list
            openList.poll();
            closedList[i][j] = true;

            // Generating all the 8 neighbors of the cell

            for (int addX = -1; addX <= 1; addX++) {
                for (int addY = -1; addY <= 1; addY++) {
                    Pair neighbour = new Pair(i + addX, j + addY);
                    if (isValid(grid, rows, cols, neighbour)) {
                        if (cellDetails[neighbour.first] == null) {
                            cellDetails[neighbour.first] = new Cell[cols];
                        }
                        if (cellDetails[neighbour.first][neighbour.second] == null) {
                            cellDetails[neighbour.first][neighbour.second] = new Cell();
                        }

                        if (isDestination(neighbour, dest)) {
                            cellDetails[neighbour.first][neighbour.second].parent = new Pair(i, j);
                            System.out.println("The destination cell is found");
                            return tracePath(cellDetails, rows, cols, dest);
                        } else if (!closedList[neighbour.first][neighbour.second] && isUnBlocked(grid, rows, cols, neighbour)) {
                            double gNew, hNew, fNew;
                            gNew = cellDetails[i][j].g + 1.0;
                            hNew = calculateHValue(neighbour, dest);
                            fNew = gNew + hNew;

                            if (cellDetails[neighbour.first][neighbour.second].f == -1 || cellDetails[neighbour.first][neighbour.second].f > fNew) {
                                openList.add(new Details(fNew, neighbour.first, neighbour.second));

                                // Update the details of this
                                // cell
                                cellDetails[neighbour.first][neighbour.second].g = gNew;
                                //heuristic function
                                cellDetails[neighbour.first][neighbour.second].h = hNew;
                                cellDetails[neighbour.first][neighbour.second].f = fNew;
                                cellDetails[neighbour.first][neighbour.second].parent = new Pair(i, j);
                            }
                        }
                    }
                }
            }

        }
        return null;
    }

    public AAsterisk(GamePanel gp) {
        this.gp = gp;
    }

    // method to insert data in to grid
    public static int[][] readFile(String filePath) {
        int[][] ans = new int[50][90];
        int row = 0;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(" ");
                for (int i = 0; i < data.length; i++) {
                    ans[row][i] = Integer.parseInt(data[i]);
                }
                row++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return ans;
    }

    // test
    public static void main(String[] args) {


        //0: The cell is blocked
        // 1: The cell is not blocked

        int[][] grid = readFile("src/res/mapData/testmap.txt");

        System.out.println(grid[48][88]);


        // Start is the left-most upper-most corner
        Pair src = new Pair(13, 54);
        //(8, 0);

        // Destination is the right-most bottom-most corner
        Pair dest = new Pair(47, 88);

        AAsterisk app = new AAsterisk(new GamePanel());
        System.out.println(app.aStarSearch(grid, grid.length, grid[0].length, src, dest));
    }
}
