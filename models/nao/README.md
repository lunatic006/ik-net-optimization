Download the meshes from here:

https://github.com/ros-naoqi/nao_meshes_installer

You'll have to update the .dae files to point to the texture file. For example, `HeadPitch.dae` should include
```
  <library_images>
    <image id="textureNAO_png_002" name="textureNAO_png_002">
      <init_from>../../texture/textureNAO.png</init_from>
    </image>
  </library_images>
```
Then, run `dae_to_gltf.sh` to convert the files to a format drake can use.

You'll also have to run `stl_to_obj.sh` to convert the collision geometries to a format drake can use.