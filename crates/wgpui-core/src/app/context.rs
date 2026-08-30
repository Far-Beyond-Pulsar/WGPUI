use super::{App, Entity, EntityId, Subscription, Task};
pub struct Context<T> {
    entity: Entity<T>,
    notified: bool,
}
impl<T> Context<T> {
    pub(crate) fn new(entity: Entity<T>) -> Self {
        Self {
            entity,
            notified: false,
        }
    }
    pub fn entity(&self) -> Entity<T> {
        self.entity.clone()
    }
    /// Mark this entity dirty. `update` delivers the notification after its
    /// mutable borrow ends, so observers may safely read the entity.
    pub fn notify(&mut self) {
        self.notified = true;
    }
    pub fn spawn<F, Fut, R>(&self, make: F) -> Task<R>
    where
        F: FnOnce(super::WeakEntity<T>, Context<T>) -> Fut,
        Fut: std::future::Future<Output = R> + 'static,
        R: 'static,
    {
        self.entity.app().spawn(make(
            self.entity.downgrade(),
            Context::new(self.entity.clone()),
        ))
    }
    pub fn observe(&self, callback: impl Fn(EntityId) + 'static) -> Subscription {
        self.entity.observe(callback)
    }
    pub fn app(&self) -> App {
        self.entity.app()
    }
}
