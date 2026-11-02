import batch3Dregistration.B3dParameters;
import batch3Dregistration.Batch_3D_Registration;
import io.scif.services.DatasetIOService;
import org.scijava.ops.image.transform.Transforms;
import net.imagej.Dataset;
import net.imagej.ops.OpService;
import net.imglib2.FinalDimensions;
import net.imglib2.type.numeric.real.FloatType;
import org.jruby.RubyProcess;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.scijava.Context;
import org.junit.Test;
import org.scijava.command.CommandService;
import org.scijava.legacy.service.OpEnvironmentService;
import org.scijava.ops.image.*;
import org.scijava.ops.api.OpEnvironment;

import java.io.File;

public class BasicFunctionalityTest {

    private static Context context;

    @BeforeClass
    public static void setUp() {
        context = new Context();

    }

    @AfterClass
    public static void tearDown() {
        context.dispose();
    }

    @Test
    public void basicRegistrationTest() throws Exception{
        try (Context context = new Context()) {
            DatasetIOService io = context.service(DatasetIOService.class);
            CommandService cs = context.service(CommandService.class);
            OpEnvironmentService ops = context.getService(OpEnvironmentService.class);


            File reference = new File(Batch_3D_Registration.class.getResource("/ref,mri-stack.tif").getFile());
            File moving = new File(Batch_3D_Registration.class.getResource("/mri-stack-rotated60right.tif").getFile());

            System.out.println(ops.env().help("transform"));

//            cs.run(Batch_3D_Registration.class, true,
//                    "referenceFile", reference,
//                    "correlationChannel", 1,
//                    "maskFile", null,
//                    "resolutionChoice", B3dParameters.Resolutions.VeryLow,
//                    "movingFiles", new File[]{moving},
//                    "thresholdValue", 0,
//                    "saveFolder", new File("../testOut/")
//            ).get();


        }
    }
}
