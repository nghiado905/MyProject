package objects;

import algorithm.AAsterisk;
import entity.Player;
import main.GamePanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.io.File;
import java.io.IOException;

public class Chest extends SuperObject {
    private boolean correct;

    public Chest(GamePanel gp) {
        this.gp = gp;
        setDefault();
        setImage();
    }

    @Override
    public void setDefault() {
        int x = 88;
        int y = 48;
        name = "chest";
        worldX = x * GamePanel.UNIT_SIZE;
        worldY = y * GamePanel.UNIT_SIZE;
        isCollected = false;
        attainable = false;
        correct = true;
    }

    @Override
    void setImage() {
        try {
            image = ImageIO.read(new File("src/res/objects/chest.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void changeInformation(String name, int x, int y, boolean isCollected, boolean attainable) {
        this.name = name;
        this.x = x;
        this.y = y;
        this.worldX = x * GamePanel.UNIT_SIZE;
        this.worldY = y * GamePanel.UNIT_SIZE;
        this.isCollected = isCollected;
        this.attainable = attainable;
    }

    public void checkChest(Player player) {
        if (this.name.equals("suggestion") && player.isKeyEnable()) {
            int ans = -1;
            if (correct) {
                ans = JOptionPane.showOptionDialog(null, "What is the best route finding algorithm?", "Question", JOptionPane.DEFAULT_OPTION, JOptionPane.QUESTION_MESSAGE, null, new String[]{"DIJKSTRA", "A*", "DFS", "BFS"}, "");
                correct = false;
                gp.keyHandle.right = false;
            }
            if (ans == 1) {
                try {
                    JOptionPane.showMessageDialog(null, "Congratulation you have achieved the MAGNIFYING GLASS item", "Congratulation", JOptionPane.PLAIN_MESSAGE, new ImageIcon(ImageIO.read(new File("src/res/objects/magnifyingGlass.png"))));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                gp.mapSetter.setStarRoad(gp.asterisk.aStarSearch(gp.map, GamePanel.MAX_ROW, GamePanel.MAX_COLUMN, new AAsterisk.Pair(13, 54), new AAsterisk.Pair(47, 88)));
            } else {
                JOptionPane.showMessageDialog(null, "Wrong choice");
            }
        } else if (this.name.equals("chest")) {
            gp.setGameWin(true);
            System.out.println("Win!!!");
        }
    }

    public boolean isCorrect() {
        return correct;
    }
}
