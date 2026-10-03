import batch3Dregistration.Batch_3D_Registration
import net.imagej.ImageJ
import org.scijava.command.CommandInfo
import org.scijava.module.ModuleInfo

ImageJ ij = new ImageJ();
ij.launch();

ModuleInfo batch3Dmodule = new CommandInfo(Batch_3D_Registration.class);

ij.module().addModule(batch3Dmodule);

//Dataset ref = ij.scifio().datasetIO().open("../resources/ref,mri-stack.tif");
//
//Dataset ref = ij.scifio().datasetIO().open("../resources/ref,mri-stack.tif");
//Dataset multiple = ij.scifio().datasetIO().open("../resources/mri-stack-arbitrary-multiaxis rotation.tif");

ij.module().run(batch3Dmodule,
    true,
    "referenceFile", new File("../resources/ref,mri-stack.tif"),
        "resolutionChoice", B3dParameters.Resolutions.Moderate,
        "movingFiles", new File[]{new File("../resources/mri-stack-rotated60right.tif"), new File("../resources/mri-stack-arbitrary-multiaxis rotation.tif")},
        "thresholdValue", 0.0,
        "saveFolder", new File("../testOut/")
    )
