package objects;

import entity.Player;
import main.GamePanel;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class Axe extends SuperObject {
    private boolean capableOfChop;
    public Axe(GamePanel gp) {
        this.gp = gp;
        setDefault();
        setImage();
        capableOfChop = false;
    }
    @Override
    public void setDefault() {
        x = 4;
        y = 10;
        name = "axe";
        worldX = x * GamePanel.UNIT_SIZE;
        worldY = y * GamePanel.UNIT_SIZE;
        isCollected = false;
        attainable = true;
    }
    public void changeInformation(){
        x = 42;
        y = 4;
        name = "axe1";
        worldX = x * GamePanel.UNIT_SIZE;
        worldY = y * GamePanel.UNIT_SIZE;
    }
    @Override
    void setImage() {
        try {
            image = ImageIO.read(new File("src/res/objects/axe.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public boolean isCapableOfChop() {
        return capableOfChop;
    }

    public void setCapableOfChop(boolean capableOfChop) {
        this.capableOfChop = capableOfChop;
    }
    public void chop(Player player){
        setCapableOfChop(false);
        player.useAxe(this);
    }
}
