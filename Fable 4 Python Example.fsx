#r"nuget: Fable.Core, 4.0.0-theta-001"

open Fable.Core

type IPhysicalDevice =
    abstract name: string
    abstract device_type: string

type IConfig =
    abstract list_physical_devices : string -> IPhysicalDevice array

type NDArray =
    [<Emit("$0.reshape($1...)")>]
    member this.reshape([<ParamList>] args : int[]): NDArray = nativeOnly
    [<Emit("$0.ndim")>]
    member this.ndim : int = nativeOnly
    [<Emit("$0.shape")>]
    member this.shape : int array = nativeOnly
    [<Emit("$0 / $1")>]
    member this.``/``(arg : obj) : NDArray = nativeOnly
    [<Emit("$0")>]
    static member fromArray(arg : 'a[]): NDArray = nativeOnly
    [<Emit("$0")>]
    member this.toArray<'a>() : array<'a> = nativeOnly

type IMnist =
    abstract load_data : unit -> (NDArray * NDArray) * (NDArray * NDArray)

type IDataSets =
    abstract mnist : IMnist

type IKerasTensor = IKerasTensor

type IDense = IKerasTensor -> IKerasTensor

type Layers =
    [<NamedParams(fromIndex = 0)>]
    abstract Dense :
        units:int ->
            IDense

type ISummary = ISummary

type IOptimizer = IOptimizer
type ILoss = ILoss
type IHistory = IHistory

type IOptimizers =
    [<NamedParams(fromIndex = 0)>]
    abstract SGD :
        learning_rate:float ->
            IOptimizer

type ILosses =
    [<NamedParams(fromIndex = 0)>]
    abstract SparseCategoricalCrossentropy :
        from_logits :bool ->
            ILoss

type IModel =
    abstract summary : unit -> ISummary

    [<NamedParams(fromIndex = 0)>]
    abstract compile :
        optimizer:IOptimizer *
        loss:ILoss *
        metrics: string ->
            IDense

    [<NamedParams(fromIndex = 0)>]
    abstract fit :
        x:NDArray *
        y:NDArray *
        epochs: int ->
            IHistory

    [<NamedParams(fromIndex = 0)>]
    abstract evaluate :
        x:NDArray *
        y:NDArray *
        verbose: int ->
            int

type IKeras =
    abstract datasets : IDataSets

    [<NamedParams(fromIndex = 0)>]
    abstract Input: 
        // it would be nice if this could be a param array
        shape:int[] ->
            IKerasTensor

    abstract layers : Layers

    [<NamedParams(fromIndex = 0)>]
    abstract Model : 
        inputs:IKerasTensor *
        outputs:IKerasTensor *
        name: string ->
            IModel

    abstract optimizers: IOptimizers
    abstract losses: ILosses

type ITensorFlow =
    abstract config : IConfig
    abstract keras : IKeras

[<ImportAll("tensorflow")>]
let tensorflow : ITensorFlow = nativeOnly

tensorflow.config.list_physical_devices("CPU")

let ((image_train, label_train), (image_test, label_test)) = 
   tensorflow.keras.datasets.mnist.load_data()

printfn $"Image train shape: {image_train.shape}"

let image_train_flat = (image_train.reshape [| 60000; 784 |]).``/`` 255

printfn $"Image train flat shape: {image_train_flat.shape}"

let image_test_flat = (image_test.reshape [| 10000; 784 |]).``/`` 255

let inputs = tensorflow.keras.Input(shape = [| 784 |])

let dense = tensorflow.keras.layers.Dense(units = 10)

let outputs = dense inputs

let model = tensorflow.keras.Model(inputs = inputs, outputs = outputs, name = "Digit_Recognition")

model.compile(
    tensorflow.keras.optimizers.SGD(0.1), 
    tensorflow.keras.losses.SparseCategoricalCrossentropy(true),
    "accuracy")

model.fit(image_train_flat, label_train, epochs = 1)

model.evaluate(image_test_flat, label_test, verbose = 2)

model.summary()