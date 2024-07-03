package map;

import java.awt.image.BufferedImage;

public class Tile {
    public BufferedImage image;
    public boolean isBarricade;
    public Tile(BufferedImage image, boolean isBarricade){
        this.image = image;
        this.isBarricade = isBarricade;
    }
    public Tile(){

    }
}
