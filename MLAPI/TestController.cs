using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLAPI
{
    [Route("api/[controller]")]
    [ApiController]
    public class TestController : ControllerBase
    {
        [HttpGet]
        public ActionResult Index() {
            //Create MLContext
            MLContext mlContext = new MLContext();
            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            var ModelFile = Path.GetFullPath("MLModel.mlnet");
            
            ITransformer predictionPipeline = mlContext.Model.Load(ModelFile, out predictionPipelineSchema);
            // Create PredictionEngines
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(predictionPipeline);

            // Input Data
            ModelInput inputData = new ModelInput
            {
                Col0 = "Crust is not good."
            };

            // Get Prediction
            ModelOutput prediction = predictionEngine.Predict(inputData);

            return Ok(prediction.PredictedLabel == 1 ? "good" : "bad");
        }
    }


    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"col0")]
        public string Col0 { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"col1")]
        public float Col1 { get; set; }

    }


    public class ModelOutput
    {
        [ColumnName(@"col0")]
        public float[] Col0 { get; set; }

        [ColumnName(@"col1")]
        public uint Col1 { get; set; }

        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public float PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }
}
