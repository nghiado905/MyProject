package map;

import algorithm.AAsterisk;

public class LinkDoor {
    private AAsterisk.Pair gate;
    private AAsterisk.Pair door;
    public LinkDoor(AAsterisk.Pair gate, AAsterisk.Pair door){
        this.gate = gate;
        this.door = door;
    }

    public AAsterisk.Pair getGate() {
        return gate;
    }

    public AAsterisk.Pair getDoor() {
        return door;
    }
}
