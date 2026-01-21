using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLAPI
{
    [Route("api/[controller]")]
    [ApiController]
    public class TestVoiceController : ControllerBase
    {
        [HttpGet]
        public ActionResult Index() {
            //Create MLContext
            MLContext mlContext = new MLContext();
            // Load Trained Model
            DataViewSchema predictionPipelineSchema;
            var ModelFile = Path.GetFullPath("emotionClassification.mlnet");
            
            ITransformer predictionPipeline = mlContext.Model.Load(ModelFile, out predictionPipelineSchema);
            // Create PredictionEngines
            PredictionEngine<VoiceModelInput, VoiceModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<VoiceModelInput, VoiceModelOutput>(predictionPipeline);

            // Input Data
            VoiceModelInput inputData = new VoiceModelInput
            {
                Col0 = -53.0878f,
                Col1 = 1.7838501f,
                Col2 = 0.16679822f,
                Col3 = -0.08908448f,
                Col4 = -0.4072238f,
                Col5 = -0.7186171f,
                Col6 = -0.55853546f,
                Col7 = -0.40691668f,
                Col8 = -0.2775434f,
                Col9 = -0.3254362f,
                Col10 = -0.22339426f,
                Col11 = -0.25185585f,
                Col12 = -0.21989082f,
                Col13 = 70.60686f,
                Col14 = 2.5933323f,
                Col15 = 1.0606415f,
                Col16 = 1.0527081f,
                Col17 = 0.884687f,
                Col18 = 0.8091588f,
                Col19 = 0.6836234f,
                Col20 = 0.49215502f,
                Col21 = 0.5542993f,
                Col22 = 0.44158593f,
                Col23 = 0.3581943f,
                Col24 = 0.4708691f,
                Col25 = 0.33864424f,
            };

            // Get Prediction
            VoiceModelOutput prediction = predictionEngine.Predict(inputData);

            return Ok((Emotion)prediction.PredictedLabel);
        }
    }


    public enum Emotion
    {
        neutral = 1,
        calm,
        happy,
        sad,
        angry,
        fearful,
        disgust,
        surprised,
    }

    public class VoiceModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"col0")]
        public float Col0 { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"col1")]
        public float Col1 { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"col2")]
        public float Col2 { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"col3")]
        public float Col3 { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"col4")]
        public float Col4 { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"col5")]
        public float Col5 { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"col6")]
        public float Col6 { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"col7")]
        public float Col7 { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"col8")]
        public float Col8 { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"col9")]
        public float Col9 { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"col10")]
        public float Col10 { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"col11")]
        public float Col11 { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"col12")]
        public float Col12 { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"col13")]
        public float Col13 { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"col14")]
        public float Col14 { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"col15")]
        public float Col15 { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"col16")]
        public float Col16 { get; set; }

        [LoadColumn(17)]
        [ColumnName(@"col17")]
        public float Col17 { get; set; }

        [LoadColumn(18)]
        [ColumnName(@"col18")]
        public float Col18 { get; set; }

        [LoadColumn(19)]
        [ColumnName(@"col19")]
        public float Col19 { get; set; }

        [LoadColumn(20)]
        [ColumnName(@"col20")]
        public float Col20 { get; set; }

        [LoadColumn(21)]
        [ColumnName(@"col21")]
        public float Col21 { get; set; }

        [LoadColumn(22)]
        [ColumnName(@"col22")]
        public float Col22 { get; set; }

        [LoadColumn(23)]
        [ColumnName(@"col23")]
        public float Col23 { get; set; }

        [LoadColumn(24)]
        [ColumnName(@"col24")]
        public float Col24 { get; set; }

        [LoadColumn(25)]
        [ColumnName(@"col25")]
        public float Col25 { get; set; }

        [LoadColumn(26)]
        [ColumnName(@"col26")]
        public float Col26 { get; set; }

    }


    /// <summary>
    /// model output class for EmotionClassification.
    /// </summary>
    /// 
    public class VoiceModelOutput
    {
        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public float PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }

}
