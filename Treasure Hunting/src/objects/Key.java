package objects;

import main.GamePanel;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;

public class Key extends SuperObject{
    public Key(GamePanel gp){
        this.gp = gp;

        setDefault();
        setImage();
    }
    @Override
    public void setDefault() {
        int x = 40;
        int y = 44;
        name = "key";
        worldX = x * GamePanel.UNIT_SIZE;
        worldY = y * GamePanel.UNIT_SIZE;
        isCollected = false;
        attainable = true;
    }

    @Override
    void setImage() {
        try {
            image = ImageIO.read(new File("src/res/objects/key.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
