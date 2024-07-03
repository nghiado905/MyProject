package algorithm;

import main.GamePanel;
import java.util.HashMap;

public class UnionFind {
    private GamePanel gp;
    public UnionFind(GamePanel gp) {
        this.gp = gp;
    }
    private HashMap<String, String> parent = new HashMap<>();
    public void populate() {
        for (int i = 0; i < GamePanel.MAX_ROW; i++) {
            for (int j = 0; j < GamePanel.MAX_COLUMN; j++) {
                if (!gp.mapSetter.tiles[gp.map[i][j]].isBarricade) {
                    parent.put(i + "," + j, i + "," + j);
                }
            }
        }

        for (int i = 0; i < GamePanel.MAX_ROW; i++) {
            for (int j = 0; j < GamePanel.MAX_COLUMN; j++) {
                if (!gp.mapSetter.tiles[gp.map[i][j]].isBarricade) {
                    // check 4 neighbor element
                    if (isValid(i - 1, j) && !gp.mapSetter.tiles[gp.map[i - 1][j]].isBarricade) union(i, j, i - 1, j);
                    if (isValid(i, j - 1) && !gp.mapSetter.tiles[gp.map[i][j - 1]].isBarricade) union(i, j, i, j - 1);
                    if (isValid(i + 1, j) && !gp.mapSetter.tiles[gp.map[i + 1][j]].isBarricade) union(i, j, i + 1, j);
                    if (isValid(i, j + 1) && !gp.mapSetter.tiles[gp.map[i][j + 1]].isBarricade) union(i, j, i, j + 1);
                }
            }
        }
    }
    public boolean isValid(int i, int j) {
        return i >= 0 && j >= 0 && i < GamePanel.MAX_ROW && j < GamePanel.MAX_COLUMN;
    }

    public int[] splitNumber(String number) {
        String[] data = number.split(",");
        return new int[]{Integer.parseInt(data[0]), Integer.parseInt(data[1])};
    }

    public int[] find(int i, int j) {
        String idx = i + "," + j;
        while (!parent.get(idx).equals(idx)) {
            String parentIdx = parent.get(idx);
            if (parentIdx != null) {
                parent.put(idx, parent.get(parentIdx));
                int[] temp = splitNumber(parentIdx);
                i = temp[0];
                j = temp[1];
                idx = i + "," + j;
            } else {
                throw new RuntimeException("out!!!\n");
            }
        }
        return new int[]{i, j};
    }

    public void union(int i1, int j1, int i2, int j2) {
        int[] p = find(i1, j1);
        int[] q = find(i2, j2);
        parent.put(p[0] + "," + p[1], q[0] + "," + q[1]);
    }

    public boolean isConnected(int i1, int j1, int i2, int j2) {
        int[] idx1 = find(i1, j1);
        int[] idx2 = find(i2, j2);
        return idx1[0] == idx2[0] && idx1[1] == idx2[1];
    }

    public static void main(String[] args) {
        UnionFind find = new UnionFind(new GamePanel());
        find.populate();
        System.out.println(find.isConnected(3, 3, 3, 4));
    }
}
