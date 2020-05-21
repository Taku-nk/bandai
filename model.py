'''MELデータのノイズ除去に使ったAuto Encoder'''
# tensorflow 2.0 を使用します。

# modelに渡すデータは下の方に示したnormalize関数で正規化したものです。
# またmodelに渡すデータはMaxPoolでサイズが縦横1/2になるので、あらかじめサイズを2で割り切れるように0パディングしておく。
def create_model():
    model = tf.keras.Sequential()
    
    model.add(layers.MaxPool2D()) # まず1/2 * 1/2 = 1/4に圧縮と同時にノイズを相殺。
    
    
    model.add(layers.Conv2D(8, (5, 5), padding='same')) 
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
        
    model.add(layers.Conv2D(16, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2D(32, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    
    model.add(tf.keras.layers.UpSampling2D(interpolation='bilinear')) # ここでデータを2*2=4倍に拡大
    
    
    model.add(layers.Conv2D(16, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2D(8, (5, 5), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    
    
    model.add(layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'))

    
    model.compile(optimizer='adam',
              loss = tf.keras.losses.MeanSquaredError(),
              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),             
              metrics=['mean_squared_error'])
    
    return model 

# modelに渡すデータは以下に示す関数を用いて正規化します。
def normalize(x): # 16乗根でデータを0.0 - 1.0に正規化します。
    a = 0.001
    c = 1.22
    normed = c * (x + a)**0.0625 - c * a**0.0625
    return normed

# 上の関数の逆関数です。
def denormalize(normed):
    a = 0.001
    c = 1.22
    x = ((normed + c * a**0.0625)**16) / c**16 - a
    return x


