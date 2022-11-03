pub enum String {
    Owned(Vec<u8>),
    Const { start: usize, len: usize },
}

impl String {
    pub fn new(data: std::string::String) -> Self {
        Self::Owned(data.into_bytes())
    }

    pub fn new_const(start: usize, len: usize) -> Self {
        Self::Const { start, len }
    }
}
