package map;

import entity.Player;
import main.GamePanel;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Trap {
    BufferedImage image;
    int x;
    int y;
    boolean activate;
    GamePanel gp;

    public Trap(GamePanel gp, int x, int y) {
        this.x = x * GamePanel.UNIT_SIZE;
        this.y = y * GamePanel.UNIT_SIZE;
        this.gp = gp;
        setImage();
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    private void setImage() {
        try {
            image = ImageIO.read(new File("src/res/objects/trap.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void checkTrapped(Player player) {
        if (Math.abs(player.worldX - x) <= GamePanel.UNIT_SIZE / 2 && Math.abs(player.worldY - y) <= GamePanel.UNIT_SIZE / 2) {
            activate = true;
            if (Math.abs(player.worldX - x) <= GamePanel.UNIT_SIZE / 4 && Math.abs(player.worldY - y) <= GamePanel.UNIT_SIZE / 4) {
                gp.setGameOver(true);
            }
        } else {
            activate = false;
        }
    }

    public void draw(Graphics2D g2) {
        int screenX = x - gp.player.worldX + gp.player.xScreen;
        int screenY = y - gp.player.worldY + gp.player.yScreen;

        if (x + GamePanel.UNIT_SIZE > gp.player.worldX - gp.player.xScreen
                && x - GamePanel.UNIT_SIZE < gp.player.worldX + gp.player.xScreen
                && y + GamePanel.UNIT_SIZE > gp.player.worldY - gp.player.yScreen
                && y - GamePanel.UNIT_SIZE < gp.player.worldY + gp.player.yScreen
                && activate
        ) {
            g2.drawImage(image, screenX, screenY, GamePanel.UNIT_SIZE, GamePanel.UNIT_SIZE, null);

        }
    }
}
