package objects;

import java.awt.*;
import java.awt.image.BufferedImage;

import entity.Player;
import main.GamePanel;

public abstract class SuperObject {
    int x, y;
    int worldX, worldY;
    boolean isCollected;
    boolean attainable;
    String name;
    BufferedImage image;
    GamePanel gp;
    public abstract void setDefault();
    abstract void setImage();
    public void checkCollection(Player player){
        if(attainable){
            if (!isCollected) {
                if (Math.abs(player.worldX - this.worldX) < GamePanel.UNIT_SIZE/2 && Math.abs(player.worldY - this.worldY) < GamePanel.UNIT_SIZE/2)
                {
                    isCollected = true;
                    image = null;
                    if(this instanceof Axe){
                        ((Axe)this).setCapableOfChop(true);
                    }
                }
            }
        } else {
            if (Math.abs(player.worldX - this.worldX) < GamePanel.UNIT_SIZE/2 && Math.abs(player.worldY - this.worldY) < GamePanel.UNIT_SIZE/2)
            {
                if(this instanceof Chest){
                    if(((Chest)this).isCorrect()) {
                        ((Chest) this).checkChest(player);
                    }
                }
            }
        }
    }

    public void draw(Graphics2D g2) {
        int screenX = worldX - gp.player.worldX + gp.player.xScreen;
        int screenY = worldY- gp.player.worldY + gp.player.yScreen;

        if (worldX + GamePanel.UNIT_SIZE > gp.player.worldX - gp.player.xScreen
                && worldX - GamePanel.UNIT_SIZE < gp.player.worldX + gp.player.xScreen
                && worldY + GamePanel.UNIT_SIZE > gp.player.worldY - gp.player.yScreen
                && worldY - GamePanel.UNIT_SIZE< gp.player.worldY + gp.player.yScreen
        ) {
            g2.drawImage(image, screenX, screenY, GamePanel.UNIT_SIZE, GamePanel.UNIT_SIZE, null);

        }
    }
    public boolean getIsCollected(){
        return isCollected;
    }
}
